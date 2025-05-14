"""
Migration script to help transition from the old plot structure to the new one.

This script provides a compatibility layer that helps migrate from the old
plot structure to the new modular architecture without breaking existing code.
"""

import logging
import sys
import os
import importlib

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PlotMigration')

def migrate():
    """
    Migrate from the old plot structure to the new one.
    
    This function:
    1. Checks if old plot_unit.py is in use
    2. Makes a backup if needed
    3. Creates symbolic links or imports to maintain backward compatibility
    """
    try:
        logger.info("Starting plot structure migration...")
        
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check if the old plot_unit.py is being imported
        old_module_path = os.path.join(script_dir, 'plot_unit.py')
        new_module_path = os.path.join(script_dir, 'new_plot_unit.py')
        
        if not os.path.exists(new_module_path):
            logger.error(f"New plot unit module not found at {new_module_path}")
            return False
            
        # Rename old module to backup if it exists and no backup exists yet
        backup_path = os.path.join(script_dir, 'plot_unit.py.bak')
        if os.path.exists(old_module_path) and not os.path.exists(backup_path):
            logger.info(f"Creating backup of old plot_unit.py to {backup_path}")
            os.rename(old_module_path, backup_path)
        
        # Create symbolic link or copy new module to old name
        if not os.path.exists(old_module_path):
            try:
                # Try symbolic link first (works on most systems)
                logger.info(f"Creating symbolic link from {new_module_path} to {old_module_path}")
                os.symlink(new_module_path, old_module_path)
            except (OSError, AttributeError):
                # Fall back to a simple copy if symlink fails
                logger.info(f"Symbolic link failed, copying {new_module_path} to {old_module_path}")
                import shutil
                shutil.copy2(new_module_path, old_module_path)
        
        logger.info("Migration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        return False

def update_imports():
    """
    Update import statements in relevant files.
    
    This function:
    1. Identifies files that import the old plot_unit module
    2. Updates their import statements to use the new structure
    """
    try:
        logger.info("Updating import statements in relevant files...")
        
        # Files likely to import plot_unit
        target_files = [
            'c:\\Users\\corre\\ComfyUI\\custom_nodes\\PIC-2025\\comfy\\Registry\\plot_unit_node.py',
            'c:\\Users\\corre\\ComfyUI\\custom_nodes\\PIC-2025\\comfy\\Registry\\standalone_visualization_hub.py'
        ]
        
        for file_path in target_files:
            if os.path.exists(file_path):
                logger.info(f"Updating imports in {file_path}")
                _update_file_imports(file_path)
                
        logger.info("Import updates completed")
        return True
        
    except Exception as e:
        logger.error(f"Error updating imports: {e}")
        return False

def _update_file_imports(file_path):
    """
    Update import statements in a specific file.
    
    Args:
        file_path (str): Path to the file to update
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Replace old import with new import
        old_import = 'from ...src.plot.plot_unit import PlotUnit'
        new_import = 'from ...src.plot import PlotUnit  # Updated import path'
        content = content.replace(old_import, new_import)
        
        with open(file_path, 'w') as f:
            f.write(content)
            
    except Exception as e:
        logger.error(f"Error updating file {file_path}: {e}")

if __name__ == '__main__':
    # Execute migration when run directly
    success = migrate()
    if success:
        update_imports()
