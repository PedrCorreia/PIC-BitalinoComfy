# fix_plot_unit.py - Run this script to patch the plot_unit.py file to fix the "raw" signal display issue

import os
import re

def patch_plot_unit():
    """Patch the plot_unit.py file to fix the raw signal display"""
    plot_unit_path = os.path.join('src', 'plot', 'plot_unit.py')
    
    if not os.path.exists(plot_unit_path):
        print(f"Error: Could not find {plot_unit_path}")
        return False
        
    # Read the file
    with open(plot_unit_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Find the problematic code block
    raw_view_pattern = r"def _draw_raw_view\(self\):.*?if not signals_to_show:.*?self\._draw_plot\(data, \"Raw Signal\", \(220, 180, 0\)\)"
    raw_view_match = re.search(raw_view_pattern, content, re.DOTALL)
    
    if not raw_view_match:
        print("Could not find the _draw_raw_view method in the file")
        return False
        
    # Get the matched text
    match_text = raw_view_match.group(0)
    
    # Create the replacement with proper "No signals available" message
    replacement = match_text.replace(
        'if not signals_to_show:\n            # Fall back to showing the default \'raw\' signal if no other signals\n            with self.data_lock:\n                data = np.copy(self.data.get(\'raw\', np.zeros(100)))\n            self._draw_plot(data, "Raw Signal", (220, 180, 0))',
        'if not signals_to_show:\n            # Draw "No signals available" message\n            msg = "No Raw Signals Available"\n            text = self.font.render(msg, True, self.text_color)\n            text_rect = text.get_rect(center=(self.sidebar_width + self.plot_width // 2, self.height // 2))\n            self.surface.blit(text, text_rect)'
    )
    
    # Apply the replacement
    patched_content = content.replace(match_text, replacement)
    
    # Write the patched file
    with open(plot_unit_path, 'w', encoding='utf-8') as f:
        f.write(patched_content)
        
    print(f"Successfully patched {plot_unit_path} to fix raw signal display")
    return True
    
if __name__ == "__main__":
    patch_plot_unit()
