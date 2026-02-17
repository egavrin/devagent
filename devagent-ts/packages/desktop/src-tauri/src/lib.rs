use serde::{Deserialize, Serialize};

#[cfg(debug_assertions)]
use tauri::Manager;

/// Message from the frontend to the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub message: String,
    pub mode: String, // "plan" or "act"
}

/// Streamed chunk from engine to frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    #[serde(rename = "type")]
    pub chunk_type: String, // "text", "tool_call", "tool_result", "done", "error"
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Engine status event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatus {
    pub status: String, // "ready", "busy", "error"
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Tauri command: send a chat message to the engine.
/// The actual LLM interaction happens on the frontend side via WASM/JS bridge.
/// This command is a pass-through for commands that need Rust-side processing.
#[tauri::command]
fn get_app_info() -> String {
    format!(
        "DevAgent v{} | Tauri v{}",
        env!("CARGO_PKG_VERSION"),
        tauri::VERSION
    )
}

/// Tauri command: get the working directory.
#[tauri::command]
fn get_working_directory() -> Result<String, String> {
    std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .map_err(|e| e.to_string())
}

/// Tauri command: set the working directory.
#[tauri::command]
fn set_working_directory(path: String) -> Result<(), String> {
    std::env::set_current_dir(&path).map_err(|e| format!("Failed to set directory: {}", e))
}

/// Tauri command: resolve the CLI entry point path.
/// Searches upward from the binary's working directory to find the monorepo CLI.
#[tauri::command]
fn get_cli_path() -> Result<String, String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;

    // Search for the CLI dist file relative to various possible working dirs
    let candidates = [
        // From src-tauri/ (cargo-tauri dev)
        cwd.join("../../cli/dist/index.js"),
        // From packages/desktop/
        cwd.join("../cli/dist/index.js"),
        // From monorepo root
        cwd.join("packages/cli/dist/index.js"),
    ];

    for candidate in &candidates {
        if let Ok(resolved) = candidate.canonicalize() {
            return Ok(resolved.to_string_lossy().to_string());
        }
    }

    Err(format!(
        "CLI entry point not found. Searched from: {}. Run 'bun run build' in the monorepo first.",
        cwd.display()
    ))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            get_app_info,
            get_working_directory,
            set_working_directory,
            get_cli_path,
        ])
        .setup(|_app| {
            #[cfg(debug_assertions)]
            {
                let window = _app.get_webview_window("main").unwrap();
                window.open_devtools();
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running DevAgent");
}
