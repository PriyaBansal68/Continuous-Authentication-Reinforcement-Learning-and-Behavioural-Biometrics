
import string
from typing import List, Dict
import keyboard


def list_punctuations() -> List[str]:
    return ["~",
        "!",
        "@",
        "#",
        "$",
        "%",
        "^",
        "&",
        "*",
        "(",
        ")",
        "_",
        "+",
        "-",
        "=",
        "[",
        "]",
        "{",
        "}",
        ":",
        ";",
        "'",
        '"',
        "\\",
        "|",
        "<",
        ",",
        ">",
        ".",
        "?",
        "/",
        "`",
    ]


def get_all_keys() -> List[str]:
    """Get all keys normalized name on keyboard"""
    misc_keys = ['ctrl']
    return [keyboard.normalize_name(key_name) for key_name in _get_all_keys()] + misc_keys


def _get_all_keys() -> List[str]:
    """Get all keys on keyboard"""
    return (
        list(string.ascii_letters)
        + list(string.digits)
        + ["F" + str(i) for i in range(1, 13)]
        + ["NUM" + str(i) for i in range(0, 10)]
        + list_punctuations()
        + [
            "TAB",
            "CAPSLOCK",
            "LSHIFT",
            "LCTRL",
            "LALT",
            "SPACE",
            "RALT",
            "RCTRL",
            "RSHIFT",
            "RETURN",
            "BACKSPACE",
            "LEFT",
            "RIGHT",
            "DOWN",
            "UP",
            "END",
            "ESCAPE",
            "DELETE",
            "HOME",
            "LWIN",
            "RWIN",
            "SHIFT",
            "ALT",
            "INSERT",
            "MENU",
            "NUM.",
            "NUM+",
            "NUM-",
            "NUM/",
            "NUM*",
            "NUMDOWN",
            "NUMLEFT",
            "NUMRIGHT",
            "NUMUP",
            "NUMEND",
            "NUMHOME",
            "NUMINS",
            "NUMDEL",
            "NUMPGDN",
            "NUMPGUP",
            "NUM LOCK",
            "PAGE DOWN",
            "PAGE UP",
        ]
    )

def get_key_category(key: str) -> str:
    # return key.lower()
    if key.lower() in ["a","t","h","s","i","n","r","e","l","space","return","backspace"]:
        return key.lower()
    elif key.lower() in list(string.ascii_letters):
        return "alphabet"
    elif key.lower() in list(string.digits):
        return "digit"
    elif key.lower() in list_punctuations():
        return "punctuation"
    else:
        return "others"
