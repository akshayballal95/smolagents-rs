use anyhow::{Context, Result, anyhow};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    pub command: String,
    pub args: Vec<String>,
    #[serde(default)]
    pub env: Option<HashMap<String, String>>,
}

impl ServerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.command.is_empty() {
            return Err(anyhow!("Server command cannot be empty"));
        }
        if self.args.is_empty() {
            return Err(anyhow!("Server args cannot be empty"));
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Servers {
    #[serde(flatten)]
    pub servers: HashMap<String, ServerConfig>,
}

impl Servers {
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;
        
        // If config file doesn't exist, create it with defaults
        if !config_path.exists() {
            Self::create_default_config(&config_path)?;
        }

        // Read and parse the config file
        let config_str = fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config file: {:?}", config_path))?;
        
        let servers: Servers = serde_yaml::from_str(&config_str)
            .with_context(|| "Failed to parse servers.yaml")?;

        // Validate all server configurations
        servers.validate()?;

        Ok(servers)
    }

    pub fn validate(&self) -> Result<()> {
        if self.servers.is_empty() {
            return Err(anyhow!("No servers configured in servers.yaml"));
        }

        for (name, config) in &self.servers {
            config.validate().with_context(|| format!("Invalid configuration for server '{}'", name))?;
        }

        Ok(())
    }

    fn create_default_config(path: &PathBuf) -> Result<()> {
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        // Write default configuration
        let default_config = include_str!("config/servers.yaml");
        fs::write(path, default_config)?;
        
        Ok(())
    }

    pub fn config_path() -> Result<PathBuf> {
        let proj_dirs = ProjectDirs::from("com", "lumo", "lumo-cli")
            .context("Failed to determine config directory")?;

        Ok(proj_dirs.config_dir().join("servers.yaml"))
    }

} 