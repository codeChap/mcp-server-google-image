use base64::Engine;
use rmcp::{
    ErrorData as McpError, ServerHandler, ServiceExt,
    handler::server::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::*,
    tool, tool_handler, tool_router,
    transport::stdio,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";
const MAX_IMAGE_DOWNLOAD_BYTES: u64 = 50 * 1024 * 1024; // 50 MB

// --- Config ---

#[derive(Deserialize)]
struct Config {
    api_key: String,
    #[serde(default = "default_save_dir")]
    save_dir: String,
}

fn default_save_dir() -> String {
    "/tmp/google-image".to_string()
}

static IMAGE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn load_config() -> Result<Config, Box<dyn std::error::Error>> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    let path = PathBuf::from(home)
        .join(".config")
        .join("mcp-server-google-image")
        .join("config.toml");
    let content = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "Failed to read config file: {}\n\
             Create it with your Gemini API key.\n\
             Example:\n\n\
             api_key = \"your-gemini-api-key\"\n\n\
             Error: {e}",
            path.display()
        )
    })?;
    let config: Config = toml::from_str(&content)
        .map_err(|e| format!("Failed to parse {}: {e}", path.display()))?;
    Ok(config)
}

// --- Gemini API request/response types ---

#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(rename = "generationConfig")]
    generation_config: GeminiGenerationConfig,
}

#[derive(Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum GeminiPart {
    Text {
        text: String,
    },
    InlineData {
        inline_data: GeminiInlineData,
    },
}

#[derive(Serialize)]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Serialize)]
struct GeminiGenerationConfig {
    #[serde(rename = "responseModalities")]
    response_modalities: Vec<String>,
    #[serde(rename = "imageConfig", skip_serializing_if = "Option::is_none")]
    image_config: Option<GeminiImageConfig>,
}

#[derive(Serialize)]
struct GeminiImageConfig {
    #[serde(rename = "aspectRatio", skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(rename = "imageSize", skip_serializing_if = "Option::is_none")]
    image_size: Option<String>,
}

// Response types
#[derive(Deserialize)]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    error: Option<GeminiError>,
}

#[derive(Deserialize)]
struct GeminiError {
    message: String,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiResponseContent,
}

#[derive(Deserialize)]
struct GeminiResponseContent {
    parts: Vec<GeminiResponsePart>,
}

#[derive(Deserialize)]
struct GeminiResponsePart {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    inline_data: Option<GeminiResponseInlineData>,
}

#[derive(Deserialize)]
struct GeminiResponseInlineData {
    mime_type: String,
    data: String,
}

// --- MCP tool parameter types ---

#[derive(Debug, Deserialize, JsonSchema)]
struct GenerateImageParams {
    #[schemars(description = "Text description of the desired image")]
    prompt: String,
    #[schemars(
        description = "Model to use. Options: \"gemini-2.5-flash-image\" (default, fast), \"gemini-3-pro-image-preview\" (professional quality)"
    )]
    model: Option<String>,
    #[schemars(
        description = "Aspect ratio. Options: \"1:1\" (default), \"2:3\", \"3:2\", \"3:4\", \"4:3\", \"4:5\", \"5:4\", \"9:16\", \"16:9\", \"21:9\""
    )]
    aspect_ratio: Option<String>,
    #[schemars(description = "Output resolution. Options: \"1K\" (default), \"2K\", \"4K\"")]
    image_size: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct EditImageParams {
    #[schemars(description = "URL (http/https) or local file path of the source image to edit")]
    image_url: String,
    #[schemars(description = "Natural language edit instructions")]
    prompt: String,
    #[schemars(
        description = "Model to use. Options: \"gemini-2.5-flash-image\" (default, fast), \"gemini-3-pro-image-preview\" (professional quality)"
    )]
    model: Option<String>,
    #[schemars(
        description = "Aspect ratio. Options: \"1:1\", \"2:3\", \"3:2\", \"3:4\", \"4:3\", \"4:5\", \"5:4\", \"9:16\", \"16:9\", \"21:9\""
    )]
    aspect_ratio: Option<String>,
    #[schemars(description = "Output resolution. Options: \"1K\" (default), \"2K\", \"4K\"")]
    image_size: Option<String>,
}

// --- MCP Server ---

#[derive(Clone)]
pub struct GoogleImageServer {
    api_key: String,
    save_dir: PathBuf,
    http: reqwest::Client,
    tool_router: ToolRouter<Self>,
}

impl GoogleImageServer {
    async fn call_gemini_api(
        &self,
        model: &str,
        parts: Vec<GeminiPart>,
        aspect_ratio: Option<String>,
        image_size: Option<String>,
    ) -> Result<GeminiResponse, String> {
        let has_image_config = aspect_ratio.is_some() || image_size.is_some();
        let image_config = if has_image_config {
            Some(GeminiImageConfig {
                aspect_ratio,
                image_size,
            })
        } else {
            None
        };

        let request = GeminiRequest {
            contents: vec![GeminiContent { parts }],
            generation_config: GeminiGenerationConfig {
                response_modalities: vec!["TEXT".to_string(), "IMAGE".to_string()],
                image_config,
            },
        };

        let url = format!("{GEMINI_API_BASE}/{model}:generateContent");

        let response = self
            .http
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {e}"))?;

        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "failed to read response body".to_string());
            return Err(format!("Gemini API error ({status}): {body}"));
        }

        response
            .json::<GeminiResponse>()
            .await
            .map_err(|e| format!("Failed to parse Gemini response: {e}"))
    }

    async fn fetch_image_bytes(&self, source: &str) -> Result<(Vec<u8>, String), String> {
        if source.starts_with("http://") || source.starts_with("https://") {
            let resp = self
                .http
                .get(source)
                .send()
                .await
                .map_err(|e| format!("Failed to download image: {e}"))?;

            if !resp.status().is_success() {
                return Err(format!("Failed to download image: HTTP {}", resp.status()));
            }

            if let Some(len) = resp.content_length() {
                if len > MAX_IMAGE_DOWNLOAD_BYTES {
                    return Err(format!(
                        "Image too large: {len} bytes (max {MAX_IMAGE_DOWNLOAD_BYTES})"
                    ));
                }
            }

            let bytes = resp
                .bytes()
                .await
                .map_err(|e| format!("Failed to read image bytes: {e}"))?;

            if bytes.len() as u64 > MAX_IMAGE_DOWNLOAD_BYTES {
                return Err(format!(
                    "Image too large: {} bytes (max {MAX_IMAGE_DOWNLOAD_BYTES})",
                    bytes.len()
                ));
            }

            let filename = source
                .rsplit('/')
                .next()
                .unwrap_or("image.png")
                .split('?')
                .next()
                .unwrap_or("image.png")
                .to_string();

            Ok((bytes.to_vec(), filename))
        } else {
            let bytes = tokio::fs::read(source)
                .await
                .map_err(|e| format!("Failed to read local file '{}': {e}", source))?;

            if bytes.len() as u64 > MAX_IMAGE_DOWNLOAD_BYTES {
                return Err(format!(
                    "Image too large: {} bytes (max {MAX_IMAGE_DOWNLOAD_BYTES})",
                    bytes.len()
                ));
            }

            Ok((bytes, source.to_string()))
        }
    }

    fn guess_mime_type(filename: &str) -> &'static str {
        let lower = filename.to_lowercase();
        if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
            "image/jpeg"
        } else if lower.ends_with(".webp") {
            "image/webp"
        } else if lower.ends_with(".gif") {
            "image/gif"
        } else {
            "image/png"
        }
    }

    async fn format_response(&self, resp: &GeminiResponse) -> String {
        if let Some(err) = &resp.error {
            return format!("Gemini API error: {}", err.message);
        }

        if resp.candidates.is_empty() {
            return "No candidates returned by Gemini API.".to_string();
        }

        let mut parts_out = Vec::new();
        let mut image_count = 0u32;

        for candidate in &resp.candidates {
            for part in &candidate.content.parts {
                if let Some(text) = &part.text {
                    parts_out.push(format!("Text: {text}"));
                }
                if let Some(inline) = &part.inline_data {
                    image_count += 1;
                    let ext = match inline.mime_type.as_str() {
                        "image/jpeg" => "jpg",
                        "image/webp" => "webp",
                        "image/gif" => "gif",
                        _ => "png",
                    };
                    match self.save_base64_image(&inline.data, ext).await {
                        Ok(path) => parts_out.push(format!("Image {image_count} saved to: {path}")),
                        Err(e) => parts_out.push(format!("Image {image_count} failed to save: {e}")),
                    }
                }
            }
        }

        if parts_out.is_empty() {
            return "No images or text returned.".to_string();
        }

        parts_out.join("\n")
    }

    async fn save_base64_image(&self, b64: &str, ext: &str) -> Result<String, String> {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| format!("Base64 decode error: {e}"))?;

        tokio::fs::create_dir_all(&self.save_dir)
            .await
            .map_err(|e| format!("Failed to create save dir: {e}"))?;

        let counter = IMAGE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let filename = format!("google-image-{timestamp}-{counter}.{ext}");
        let path = self.save_dir.join(&filename);

        tokio::fs::write(&path, &bytes)
            .await
            .map_err(|e| format!("Failed to write image: {e}"))?;

        Ok(path.to_string_lossy().to_string())
    }
}

#[tool_router]
impl GoogleImageServer {
    pub fn new(api_key: String, save_dir: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            api_key,
            save_dir: PathBuf::from(save_dir),
            http,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Generate an image from a text prompt using Google's Gemini image generation API")]
    async fn generate_image(
        &self,
        Parameters(params): Parameters<GenerateImageParams>,
    ) -> Result<CallToolResult, McpError> {
        let model = params.model.unwrap_or_else(|| "gemini-2.5-flash-image".to_string());

        let parts = vec![GeminiPart::Text { text: params.prompt }];

        match self
            .call_gemini_api(&model, parts, params.aspect_ratio, params.image_size)
            .await
        {
            Ok(resp) => {
                let text = self.format_response(&resp).await;
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
        }
    }

    #[tool(
        description = "Edit an existing image using natural language instructions via Google's Gemini API. Accepts an image URL or local file path."
    )]
    async fn edit_image(
        &self,
        Parameters(params): Parameters<EditImageParams>,
    ) -> Result<CallToolResult, McpError> {
        let model = params.model.unwrap_or_else(|| "gemini-2.5-flash-image".to_string());

        let (image_bytes, filename) = match self.fetch_image_bytes(&params.image_url).await {
            Ok(result) => result,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let mime_type = Self::guess_mime_type(&filename).to_string();
        let b64_data = base64::engine::general_purpose::STANDARD.encode(&image_bytes);

        let parts = vec![
            GeminiPart::Text { text: params.prompt },
            GeminiPart::InlineData {
                inline_data: GeminiInlineData {
                    mime_type,
                    data: b64_data,
                },
            },
        ];

        match self
            .call_gemini_api(&model, parts, params.aspect_ratio, params.image_size)
            .await
        {
            Ok(resp) => {
                let text = self.format_response(&resp).await;
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
        }
    }
}

#[tool_handler]
impl ServerHandler for GoogleImageServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::default(),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "mcp-server-google-image".to_string(),
                title: None,
                version: env!("CARGO_PKG_VERSION").to_string(),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Google Gemini image generation server. Use generate_image to create images from text prompts, \
                 or edit_image to modify existing images with natural language instructions."
                    .to_string(),
            ),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = load_config()?;
    let server = GoogleImageServer::new(cfg.api_key, cfg.save_dir);
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
