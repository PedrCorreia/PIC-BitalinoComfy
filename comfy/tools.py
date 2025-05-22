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

NODE_CLASS_MAPPINGS = globals().get("NODE_CLASS_MAPPINGS", {})
NODE_CLASS_MAPPINGS["PrintToolNode"] = PrintToolNode
NODE_CLASS_MAPPINGS["PrintMultiToolNode"] = PrintMultiToolNode

NODE_DISPLAY_NAME_MAPPINGS = globals().get("NODE_DISPLAY_NAME_MAPPINGS", {})
NODE_DISPLAY_NAME_MAPPINGS["PrintToolNode"] = "🛠️ Print Tool Node"
NODE_DISPLAY_NAME_MAPPINGS["PrintMultiToolNode"] = "🛠️ Print Multi Tool Node"
