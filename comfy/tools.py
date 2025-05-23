import numpy as np

class PrintToolNode:
    """
    A node that prints any input data with a user-supplied comment.
    Accepts any Python data type as input and prints it with the comment.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Info": ("ANY", {}),
                "comment": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "print_tool"
    CATEGORY = "Pedro_PIC/🛠️ Tools"
    OUTPUT_NODE = True 

    def print_tool(self, thing, comment):
        print(f"{comment} {thing}")
        return (thing,)

class PrintMultiToolNode:
    """
    A node that prints multiple optional inputs (HR, RR, is_peak, etc.) with a user-supplied comment.
    Accepts any combination of these inputs and prints them for debugging.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "comment": ("STRING", {"default": ""}),
            },
            "optional": {
                "HR": ("FLOAT", {}),
                "RR": ("FLOAT", {}),
                "is_peak": ("BOOLEAN", {}),
                "signal_id": ("STRING", {}),
                "thing": ("ANY", {}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "print_multi_tool"
    CATEGORY = "Pedro_PIC/🛠️ Tools"
    OUTPUT_NODE = True

    def print_multi_tool(self, comment, HR=None, RR=None, is_peak=None, signal_id=None, thing=None):
        msg = f"{comment}"
        if HR is not None:
            msg += f" | HR: {HR}"
        if RR is not None:
            msg += f" | RR: {RR}"
        if is_peak is not None:
            msg += f" | is_peak: {is_peak}"
        if signal_id is not None:
            msg += f" | signal_id: {signal_id}"
        if thing is not None:
            msg += f" | thing: {thing}"
        print(msg)
        return ()

class EnhancedPrintToolNode:
    """
    An improved print tool that accepts various data types as optional inputs.
    Only prints inputs that have values, providing a cleaner debugging experience.
    Supports strings, numbers, booleans, lists, dicts, and specialized structures.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "comment": ("STRING", {"default": "DEBUG:"}),
            },
            "optional": {
                "string_value": ("STRING", {"default": None}),
                "float_value": ("FLOAT", {"default": None}),
                "int_value": ("INT", {"default": None}),
                "bool_value": ("BOOLEAN", {"default": None}),
                "signal_id": ("STRING", {"default": None}),
                "heart_rate": ("FLOAT", {"default": None}),
                "is_peak": ("BOOLEAN", {"default": None}),
                "timestamp": ("FLOAT", {"default": None}),
                "custom_label": ("STRING", {"default": "custom"}),
                "custom_value": ("STRING", {"default": None}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "enhanced_print"
    CATEGORY = "Pedro_PIC/🛠️ Tools"
    OUTPUT_NODE = True

    def enhanced_print(self, comment, **kwargs):
        # Start with the comment
        message_parts = [comment]
        
        # Add each provided input with its label
        for key, value in kwargs.items():
            if value is not None:
                # Handle the special case where we have a custom label and value
                if key == "custom_label" and "custom_value" in kwargs and kwargs["custom_value"] is not None:
                    message_parts.append(f"{value}: {kwargs['custom_value']}")
                # Only print other values if they're not None and not the custom_value (handled above)
                elif key != "custom_value":
                    # Format the key by replacing underscores with spaces and capitalizing
                    formatted_key = key.replace('_', ' ').title()
                    message_parts.append(f"{formatted_key}: {value}")
        
        # Join all parts with a separator and print
        print(" | ".join(message_parts))
        return ()

NODE_CLASS_MAPPINGS = globals().get("NODE_CLASS_MAPPINGS", {})
NODE_CLASS_MAPPINGS["PrintToolNode"] = PrintToolNode
NODE_CLASS_MAPPINGS["PrintMultiToolNode"] = PrintMultiToolNode
NODE_CLASS_MAPPINGS["EnhancedPrintToolNode"] = EnhancedPrintToolNode

NODE_DISPLAY_NAME_MAPPINGS = globals().get("NODE_DISPLAY_NAME_MAPPINGS", {})
NODE_DISPLAY_NAME_MAPPINGS["PrintToolNode"] = "🛠️ Print Tool Node"
NODE_DISPLAY_NAME_MAPPINGS["PrintMultiToolNode"] = "🛠️ Print Multi Tool Node"
NODE_DISPLAY_NAME_MAPPINGS["EnhancedPrintToolNode"] = "🛠️ Enhanced Print Tool"
