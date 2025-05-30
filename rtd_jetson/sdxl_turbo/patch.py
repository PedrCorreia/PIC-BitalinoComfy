import os
import sys
from pathlib import Path

# Define cache directory path
CACHE_DIR = "/media/lugo/data/sd_card_cache"

# IMPORTANT: Set cache paths BEFORE importing huggingface_hub
os.environ["HF_HOME"] = f"{CACHE_DIR}/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{CACHE_DIR}/huggingface"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_DIR}/huggingface"
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/huggingface"
os.environ["XDG_CACHE_HOME"] = f"{CACHE_DIR}/cache"
os.environ["TORCH_HOME"] = f"{CACHE_DIR}/torch"

# Create directories if they don't exist
for dir_path in [f"{CACHE_DIR}/huggingface",
                 f"{CACHE_DIR}/cache",
                 f"{CACHE_DIR}/torch",
                 f"{CACHE_DIR}/diffusers"]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Now import huggingface_hub
import huggingface_hub

print(f"Using huggingface_hub version: {huggingface_hub.__version__}")
print(f"HF_HOME set to: {os.environ.get('HF_HOME')}")
print(f"HUGGINGFACE_HUB_CACHE set to: {os.environ.get('HUGGINGFACE_HUB_CACHE')}")

# Add the missing OfflineModeIsEnabled class
if not hasattr(huggingface_hub.utils, 'OfflineModeIsEnabled'):
    class OfflineModeIsEnabled(Exception):
        """
        Compatibility patch for older huggingface_hub versions.
        In newer versions, this exception is raised when attempting to make a network request in offline mode.
        """
        pass
    
    # Apply the monkey patch
    huggingface_hub.utils.OfflineModeIsEnabled = OfflineModeIsEnabled
    print("Added compatibility patch for huggingface_hub.utils.OfflineModeIsEnabled")

# Patch the file_download module to use our cache path
def patch_hf_file_download():
    try:
        # Try to modify the _cache_path function if it exists
        if hasattr(huggingface_hub.file_download, '_cache_path'):
            original_cache_path = huggingface_hub.file_download._cache_path
            def patched_cache_path(*args, **kwargs):
                return f"{CACHE_DIR}/huggingface"
            huggingface_hub.file_download._cache_path = patched_cache_path
            print("Patched huggingface_hub.file_download._cache_path")
    except:
        print("Failed to patch huggingface_hub.file_download._cache_path")

# Apply the file_download patch
patch_hf_file_download()

# Force offline mode to prevent unexpected downloads
os.environ["HF_HUB_OFFLINE"] = "1"

# Add cached_download for older diffusers versions
if not hasattr(huggingface_hub, 'cached_download'):
    def cached_download(*args, **kwargs):
        """Compatibility function for older diffusers versions"""
        print("Redirecting cached_download to hf_hub_download")
        return huggingface_hub.hf_hub_download(*args, **kwargs)
    
    # Add the compatibility function
    huggingface_hub.cached_download = cached_download
    print("Added compatibility patch for huggingface_hub.cached_download")

# Original JetsonCacheManager code with direct instantiation at module level
class JetsonCacheManager:
    def __init__(self, cache_path=CACHE_DIR):
        self.cache_path = cache_path
        self.setup_cache_dirs()
        self.redirect_environment_vars()
        print(f"JetsonCacheManager initialized with cache path: {cache_path}")
    
    def setup_cache_dirs(self):
        dirs = ["huggingface", "cache", "torch", "diffusers"]
        for dir_name in dirs:
            Path(f"{self.cache_path}/{dir_name}").mkdir(parents=True, exist_ok=True)
        
    def redirect_environment_vars(self):
        os.environ["HF_HOME"] = f"{self.cache_path}/huggingface"
        os.environ["HUGGINGFACE_HUB_CACHE"] = f"{self.cache_path}/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = f"{self.cache_path}/huggingface"
        os.environ["HF_DATASETS_CACHE"] = f"{self.cache_path}/huggingface"
        os.environ["XDG_CACHE_HOME"] = f"{self.cache_path}/cache"
        os.environ["TORCH_HOME"] = f"{self.cache_path}/torch"

# Initialize cache manager
cache_manager = JetsonCacheManager()
print("JetsonCacheManager initialized and cache directories set up.")