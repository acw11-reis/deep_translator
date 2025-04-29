#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Translator & Rephraser Pro

A desktop application using Tkinter for translating text between languages
and rephrasing text using various AI APIs (DeepL, OpenAI, DeepSeek).
Features include:
- Translation between supported languages.
- Rephrasing of source text (using LLMs or a multi-translation technique for DeepL).
- Clipboard integration via hotkey (Ctrl+C+C).
- Translation history management (load, save, merge, clear).
- API key configuration editor.
- System tray icon for background operation.

Requires: pyperclip, pystray, Pillow (PIL), keyboard, openai, requests, deepl
"""

import pyperclip
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from pystray import Icon, MenuItem, Menu
from PIL import Image
import keyboard
from datetime import datetime
from openai import OpenAI
import json
import requests
import sys
import signal
import os
import deepl
import traceback
from tkinter import ttk, font as tkFont, messagebox
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil
from tkinter import filedialog
import subprocess
import re # Added for cleaning rephrase results


# --- Supported Languages ---
# Dictionary mapping API providers to their supported languages and codes
supported_languages = {
    "DeepL": {
        "Turkish": "TR", "German": "DE", "French": "FR", "Spanish": "ES",
        "Italian": "IT", "Portuguese": "PT-PT", "Dutch": "NL",
        "Polish": "PL", "Russian": "RU", "Japanese": "JA", "Chinese (simplified)": "ZH",
        "English": "EN-GB", "English-GB": "EN-GB", "English-US": "EN-US"
    },
    "OpenAI": {
        "Turkish": "Turkish", "German": "German", "French": "French", "Spanish": "Spanish",
        "Italian": "Italian", "Portuguese": "Portuguese", "Dutch": "Dutch",
        "Polish": "Polish", "Russian": "Russian", "Japanese": "Japanese", "Chinese": "Chinese",
        "English": "English"
    },
    "DeepSeek": {
        "Turkish": "Turkish", "German": "German", "French": "French", "Spanish": "Spanish",
        "Italian": "Italian", "Portuguese": "Portuguese", "Dutch": "Dutch",
        "Polish": "Polish", "Russian": "Russian", "Japanese": "Japanese", "Chinese": "Chinese",
        "English": "English"
    }
}

# --- Configuration Files ---
HISTORY_FILE = "translation_history.xml"
CONFIG_FILE = "config.xml"

# --- API Keys (Loaded from config file) ---
OPENAI_API_KEY = None
DEEPSEEK_API_KEY = None
DEEPL_API_KEY = None

# --- API Clients (Initialised after loading keys) ---
openai_client = None
deepl_translator = None

# --- Global Variables for GUI and State ---
window = None
original_textbox = None
translated_textbox = None
rephrased_textbox = None
rephrase_button = None
translate_button = None
reverse_translate_button = None
history_button = None
style_dropdown = None
history_window = None
style_var = None
api_provider_var = None
target_language_var = None
source_language_var = None
last_selected_text = ""       # Stores the source text for rephrasing
last_translation = ""         # Stores the last translation result
history_data = []             # In-memory list of history items
font_size = 13                # Default font size for text boxes
tray_icon = None
hotkey_processing = False     # Flag to prevent concurrent hotkey actions
api_dropdown = None
history_lock = threading.Lock() # Lock for history file access
config_lock = threading.Lock()  # Lock for config file access
config_window = None
source_language_dropdown = None
target_language_dropdown = None

# Global variables to store the direction of the last translation for history
last_history_source_language = ""
last_history_target_language = ""

# --- Style Options for Rephrasing (Used by LLMs) ---
style_options = {
    "Simple English": "in simple and clear English",
    "Business English": "in professional business English",
    "Casual English": "in casual conversational English"
}

def get_selected_style():
    """Returns the description for the currently selected rephrasing style."""
    if style_var:
        return style_options.get(style_var.get(), style_options["Simple English"])
    return style_options["Simple English"] # Default style

# --- Prompt Generation Functions ---
def full_prompt(text, source_language, target_language):
    """
    Generates the prompt for the 'Translate & Rephrase' action for LLMs.
    Asks the LLM to first translate to the target language, then rephrase
    the *original* source text according to the selected style.
    Specifies the required response format.
    """
    style = get_selected_style()
    provider = api_provider_var.get() if api_provider_var else "DeepSeek"
    if provider in ["OpenAI", "DeepSeek"]:
        return f'''
Given the following text in {source_language}:

"{text}"

1. Translate it into {target_language}.
2. Rephrase the ORIGINAL {source_language} text {style}.

Respond ONLY in this format:
{target_language} Translation: ...
{source_language} Rephrased: ...
'''
    # DeepL does not use this prompt format directly.
    return ""

def rephrase_prompt(text, source_language):
    """
    Generates the prompt for the 'Rephrase Again' action for LLMs.
    Asks the LLM to rephrase the source text in 5 different ways,
    according to the selected style, returning a numbered list.
    """
    style = get_selected_style()
    return f"Rephrase the following text in {source_language} {style} in 5 different ways. Provide the results in a numbered list (1. ..., 2. ..., etc.) without extra comments:\n\n{text}"

def translate_to_source_prompt(text, source_language):
    """
    Generates the prompt for the 'Translate Back to Source' action for LLMs.
    Asks the LLM to translate the provided text into the specified source language.
    """
    return f"Translate the following text into {source_language}:\n\n{text}"

# --- DeepL Helper Functions ---
def get_deepl_source_code(lang: str) -> str:
    """
    Gets the DeepL source language code.
    Maps English variants ('English', 'English-GB', 'English-US') to 'EN'
    as required by DeepL's source_lang parameter.
    For other languages, uses the code from the supported_languages dictionary.
    """
    code = supported_languages["DeepL"].get(lang, lang)
    # --- Mappings for source_lang ---
    # English variants -> EN
    if code in ["EN-GB", "EN-US"]:
        return "EN"
    # Brazilian Portuguese -> PT
    elif code == "PT-PT": # Check if other Portuguese variants exist in your dict
        return "PT"
    # Add mappings for other variants if needed (e.g., Chinese)
    # --- End Mappings ---
    else:
        # For base languages (TR, DE, FR, etc.) or variants accepted as source, return the code directly.
        return code

def get_deepl_target_code(lang: str) -> str:
    """
    Gets the DeepL target language code from the dictionary.
    Target codes can include regional variations (e.g., EN-GB, EN-US, PT-BR).
    """
    code = supported_languages["DeepL"].get(lang, lang)
    return code

# --- API Interaction ---
def ask_ai(prompt=None, original_text_for_deepl=None, target_lang_for_deepl=None, source_lang_for_deepl=None):
    """
    Sends a request to the selected AI API provider.

    Handles requests for OpenAI, DeepSeek, and DeepL.
    Returns the API response content or a dictionary (for DeepL)
    or an error string prefixed with '__ERROR__::'.

    Args:
        prompt (str, optional): The prompt for LLMs (OpenAI, DeepSeek).
        original_text_for_deepl (str, optional): Text for DeepL translation.
        target_lang_for_deepl (str, optional): Target language code for DeepL.
        source_lang_for_deepl (str, optional): Source language code for DeepL.

    Returns:
        str | dict: API response or error string.
    """
    provider = api_provider_var.get() if api_provider_var else "DeepSeek"
    try:
        if provider == "OpenAI":
            if not openai_client: return "__ERROR__::OpenAI API Key not configured."
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                timeout=30.0 )
            return response.choices[0].message.content.strip()

        elif provider == "DeepSeek":
            if not DEEPSEEK_API_KEY or "YOUR_DEEPSEEK_API_KEY" in DEEPSEEK_API_KEY:
                return "__ERROR__::DeepSeek API Key not configured."
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
            payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}
            r = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, data=json.dumps(payload), timeout=30)
            r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            response_json = r.json()
            if "choices" in response_json and response_json["choices"]:
                return response_json["choices"][0]["message"]["content"]
            else: return f"__ERROR__::DeepSeek returned unexpected response: {response_json}"

        elif provider == "DeepL":
            if not deepl_translator: return "__ERROR__::DeepL API Key not configured."
            # DeepL requires specific text and language codes, not a general prompt.
            if target_lang_for_deepl and original_text_for_deepl:
                result = deepl_translator.translate_text(
                    original_text_for_deepl,
                    source_lang=source_lang_for_deepl, # Can be None for auto-detect
                    target_lang=target_lang_for_deepl
                )
                # Return a dictionary for easier processing in calling functions
                return {"translated": result.text, "rephrased": None} # DeepL doesn't rephrase
            else: return "__ERROR__::DeepL requires text and target language for translation."

        else: return "__ERROR__::Invalid API Provider selected."

    # --- Exception Handling ---
    except deepl.AuthorizationException:
        return "__ERROR__::DeepL Authentication Failed. Please check your API key."
    except deepl.DeepLException as e:
        # Catch specific DeepL errors (e.g., quota exceeded, invalid language)
        return f"__ERROR__::DeepL API Error: {str(e)}"
    except requests.exceptions.Timeout:
        return f"__ERROR__::Network timeout connecting to {provider} API."
    except requests.exceptions.RequestException as e:
        # Catch other connection errors (DNS issues, connection refused, etc.)
        return f"__ERROR__::Network error connecting to {provider}: {str(e)}"
    except Exception as e:
        # Catch any other unexpected errors
        traceback.print_exc() # Log the full traceback for debugging
        return f"__ERROR__::An unexpected error occurred ({type(e).__name__}): {str(e)}"


# --- Configuration File Handling ---
def prettify_xml(elem):
    """Returns a pretty-printed XML string for the Element."""
    try:
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        # Use toprettyxml for indentation, decode, then clean up empty lines
        pretty_xml_bytes = reparsed.toprettyxml(indent="  ", encoding='utf-8')
        pretty_xml_str = pretty_xml_bytes.decode('utf-8')
        lines = pretty_xml_str.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        cleaned_xml_str = "\n".join(non_empty_lines)
        return cleaned_xml_str.encode('utf-8')
    except Exception:
        # Fallback to basic tostring if prettify fails
        return ET.tostring(elem, 'utf-8')
# --- Context Menu Helper ---
def add_text_widget_context_menu(text_widget):
    """Adds a standard Copy/Cut/Paste/Select All context menu to a tk.Text widget."""
    context_menu = tk.Menu(text_widget, tearoff=0)

    def copy_action(event=None):
        if text_widget.tag_ranges(tk.SEL): text_widget.event_generate("<<Copy>>")
    def cut_action(event=None):
        if text_widget.tag_ranges(tk.SEL): text_widget.event_generate("<<Cut>>")
    def paste_action(event=None):
        try:
            # Generate paste event only if clipboard seems to have content
            # This might not be perfect but prevents pasting empty strings
            if text_widget.tk.call('clipboard', 'get'):
                 text_widget.event_generate("<<Paste>>")
        except tk.TclError: pass # Clipboard empty or non-text
    def select_all_action(event=None):
        text_widget.tag_add(tk.SEL, "1.0", tk.END); text_widget.focus_set(); return "break"

    context_menu.add_command(label="Cut", command=cut_action)
    context_menu.add_command(label="Copy", command=copy_action)
    context_menu.add_command(label="Paste", command=paste_action)
    context_menu.add_separator()
    context_menu.add_command(label="Select All", command=select_all_action)

    def show_popup_menu(event):
        # Update menu state based on selection and clipboard
        try:
            has_selection = bool(text_widget.tag_ranges(tk.SEL))
            context_menu.entryconfig("Cut", state=tk.NORMAL if has_selection else tk.DISABLED)
            context_menu.entryconfig("Copy", state=tk.NORMAL if has_selection else tk.DISABLED)
        except tk.TclError: # Widget might be destroyed
            context_menu.entryconfig("Cut", state=tk.DISABLED)
            context_menu.entryconfig("Copy", state=tk.DISABLED)
        try:
            has_clipboard = bool(text_widget.tk.call('clipboard', 'get'))
            context_menu.entryconfig("Paste", state=tk.NORMAL if has_clipboard else tk.DISABLED)
        except tk.TclError:
            context_menu.entryconfig("Paste", state=tk.DISABLED)
        # Display menu
        try: context_menu.tk_popup(event.x_root, event.y_root)
        finally: context_menu.grab_release()

    # Bind right-click event
    # Use platform-specific binding if needed, Button-3 is common
    text_widget.bind("<Button-3>", show_popup_menu) # Windows/Linux
    text_widget.bind("<Button-2>", show_popup_menu) # macOS (sometimes)

    # Bind Ctrl+A for Select All
    text_widget.bind("<Control-a>", select_all_action)
    text_widget.bind("<Control-A>", select_all_action) # Case-insensitive
def create_default_config():
    """Creates a default config.xml file if it doesn't exist."""
    print(f"Creating default configuration file: {CONFIG_FILE}")
    with config_lock:
        try:
            root = ET.Element('config')
            api_keys_elem = ET.SubElement(root, 'api_keys')
            ET.SubElement(api_keys_elem, 'openai').text = "YOUR_OPENAI_KEY_HERE"
            ET.SubElement(api_keys_elem, 'deepseek').text = "YOUR_DEEPSEEK_KEY_HERE"
            ET.SubElement(api_keys_elem, 'deepl').text = "YOUR_DEEPL_KEY_HERE" 
            xml_bytes = prettify_xml(root)
            with open(CONFIG_FILE, "wb") as f:
                f.write(xml_bytes)
            print(f"Default config file '{CONFIG_FILE}' created. Please edit it with your API keys.")
            # Inform user via messagebox
            messagebox.showinfo("Config File Created",
                                f"'{CONFIG_FILE}' created.\nPlease add your API keys and use 'Reload Config & Keys' from the tray menu.",
                                icon='info')
        except Exception as e:
            print(f"Error creating default config file: {e}")
            traceback.print_exc()
            messagebox.showerror("Config Error", f"Could not create default config file:\n{e}")

def load_api_keys():
    """Loads API keys from the config.xml file into global variables."""
    global OPENAI_API_KEY, DEEPSEEK_API_KEY, DEEPL_API_KEY
    loaded_keys_count = 0
    if not os.path.exists(CONFIG_FILE):
        create_default_config() # Create if missing
        OPENAI_API_KEY, DEEPSEEK_API_KEY, DEEPL_API_KEY = None, None, None
        return loaded_keys_count # No keys loaded yet

    print(f"Loading API keys from {CONFIG_FILE}...")
    with config_lock:
        try:
            tree = ET.parse(CONFIG_FILE)
            root = tree.getroot()
            keys_root = root.find('api_keys')
            if keys_root is None:
                print(f"Warning: <api_keys> tag not found in {CONFIG_FILE}. Recreating default.")
                create_default_config() # Recreate if malformed
                OPENAI_API_KEY, DEEPSEEK_API_KEY, DEEPL_API_KEY = None, None, None
                return loaded_keys_count

            # Load keys, check if they are placeholders
            openai_key = keys_root.findtext('openai', default="").strip()
            deepseek_key = keys_root.findtext('deepseek', default="").strip()
            deepl_key = keys_root.findtext('deepl', default="").strip()

            if openai_key and "YOUR_OPENAI_KEY_HERE" not in openai_key:
                OPENAI_API_KEY = openai_key; loaded_keys_count += 1
            else: OPENAI_API_KEY = None
            if deepseek_key and "YOUR_DEEPSEEK_KEY_HERE" not in deepseek_key:
                DEEPSEEK_API_KEY = deepseek_key; loaded_keys_count += 1
            else: DEEPSEEK_API_KEY = None
            if deepl_key and "YOUR_DEEPL_KEY_HERE" not in deepl_key:
                DEEPL_API_KEY = deepl_key; loaded_keys_count += 1
            else: DEEPL_API_KEY = None

            print(f"  - OpenAI Key {'Loaded' if OPENAI_API_KEY else 'Missing/Placeholder'}.")
            print(f"  - DeepSeek Key {'Loaded' if DEEPSEEK_API_KEY else 'Missing/Placeholder'}.")
            print(f"  - DeepL Key {'Loaded' if DEEPL_API_KEY else 'Missing/Placeholder'}.")
            return loaded_keys_count

        except ET.ParseError as e:
            print(f"Error parsing {CONFIG_FILE}: {e}")
            messagebox.showerror("Config Error", f"Error parsing config file:\n{e}\nA default file might be created.")
            try: os.remove(CONFIG_FILE) # Remove corrupted file
            except OSError: pass
            create_default_config()
            OPENAI_API_KEY, DEEPSEEK_API_KEY, DEEPL_API_KEY = None, None, None
            return 0
        except Exception as e:
            print(f"Error loading {CONFIG_FILE}: {e}")
            traceback.print_exc()
            messagebox.showerror("Config Error", f"Error loading config file:\n{e}")
            OPENAI_API_KEY, DEEPSEEK_API_KEY, DEEPL_API_KEY = None, None, None
            return 0

def reinitialize_clients():
    """Re-initialises the API client objects based on the currently loaded keys."""
    global openai_client, deepl_translator
    print("Re-initialising API clients...")
    # OpenAI
    if OPENAI_API_KEY:
        try: openai_client = OpenAI(api_key=OPENAI_API_KEY); print("  - OpenAI client re-initialised.")
        except Exception as e: print(f"  - Failed re-init OpenAI: {e}"); openai_client = None
    else: openai_client = None; print("  - OpenAI client set to None (no key).")
    # DeepL
    if DEEPL_API_KEY:
        try: deepl_translator = deepl.Translator(DEEPL_API_KEY); print("  - DeepL translator re-initialised.")
        except ImportError: print("  - Error: 'deepl' library not found."); deepl_translator = None
        except Exception as e: print(f"  - Failed re-init DeepL: {e}"); deepl_translator = None
    else: deepl_translator = None; print("  - DeepL translator set to None (no key).")

def update_gui_after_reload():
    """Updates the GUI elements (API dropdown, language lists) after config reload."""
    global api_dropdown, api_provider_var, window, target_language_var, source_language_var, source_language_dropdown, target_language_dropdown
    if not all([window, window.winfo_exists(), api_dropdown, api_provider_var,
                target_language_var, source_language_var, source_language_dropdown,
                target_language_dropdown]):
        # If GUI elements aren't ready, do nothing.
        return

    print("Updating GUI after config reload...")
    # Determine available providers based on loaded clients/keys
    available_providers = []
    default_provider = "No APIs Configured"
    if deepl_translator: available_providers.append("DeepL"); default_provider = "DeepL"
    if DEEPSEEK_API_KEY: available_providers.append("DeepSeek") # Check key existence
    if openai_client: available_providers.append("OpenAI") # Check client object

    # Set a sensible default if DeepL isn't available but others are
    if default_provider == "No APIs Configured" and available_providers:
        if "DeepSeek" in available_providers: default_provider = "DeepSeek"
        elif "OpenAI" in available_providers: default_provider = "OpenAI" # Or the first available
        else: default_provider = available_providers[0]

    # Update API dropdown
    if not available_providers:
        available_providers.append("No APIs Configured")
    api_dropdown['values'] = available_providers
    current_selection = api_provider_var.get()

    # Set dropdown state and selection
    if not available_providers or default_provider == "No APIs Configured":
        api_provider_var.set("No APIs Configured")
        api_dropdown.config(state="disabled")
    elif current_selection not in available_providers:
        api_provider_var.set(default_provider)
        api_dropdown.config(state="readonly")
    else:
        # Keep current selection if it's still valid
        api_dropdown.config(state="readonly")

    # Update language dropdowns based on the selected provider
    provider = api_provider_var.get()
    lang_keys = list(supported_languages.get(provider, {}).keys())

    if lang_keys:
        # Set default languages if provider changed or was invalid
        if source_language_var.get() not in lang_keys: source_language_var.set("English")
        if target_language_var.get() not in lang_keys: target_language_var.set("Turkish")
        source_language_dropdown['values'] = lang_keys
        target_language_dropdown['values'] = lang_keys
        source_language_dropdown.config(state="readonly")
        target_language_dropdown.config(state="readonly")
    else:
        # No languages for this provider (or "No APIs")
        source_language_var.set("")
        target_language_var.set("")
        source_language_dropdown['values'] = []
        target_language_dropdown['values'] = []
        source_language_dropdown.config(state="disabled")
        target_language_dropdown.config(state="disabled")

    update_button_states() # Update button text and states
    print("GUI update complete.")

def reload_config_and_clients():
    """Handles the 'Reload Config & Keys' action."""
    print("-" * 30 + "\nReloading Configuration..." + "-" * 30)
    keys_loaded = load_api_keys()
    reinitialize_clients()
    if window:
        # Schedule GUI update on the main thread
        window.after(0, update_gui_after_reload)
    messagebox.showinfo("Reload Complete", f"{keys_loaded} API key(s) reloaded.\nAPI list and clients updated.", icon='info')
    print("Reload process finished.\n" + "-" * 60)

def save_api_keys_to_xml(openai_key, deepseek_key, deepl_key):
    """Saves the provided API keys to the config.xml file."""
    print(f"Saving API keys to {CONFIG_FILE}...")
    with config_lock:
        try:
            root = ET.Element('config')
            api_keys_elem = ET.SubElement(root, 'api_keys')
            # Use placeholders if keys are empty
            ET.SubElement(api_keys_elem, 'openai').text = openai_key if openai_key and openai_key.strip() else "YOUR_OPENAI_KEY_HERE"
            ET.SubElement(api_keys_elem, 'deepseek').text = deepseek_key if deepseek_key and deepseek_key.strip() else "YOUR_DEEPSEEK_KEY_HERE"
            ET.SubElement(api_keys_elem, 'deepl').text = deepl_key if deepl_key and deepl_key.strip() else "YOUR_DEEPL_KEY_HERE_OR_FREE_KEY:fx"
            xml_bytes = prettify_xml(root)
            with open(CONFIG_FILE, "wb") as f:
                f.write(xml_bytes)
            print(f"API Keys successfully saved to {CONFIG_FILE}")
            return True
        except Exception as e:
            print(f"Error saving config file '{CONFIG_FILE}': {e}")
            traceback.print_exc()
            messagebox.showerror("Config Save Error", f"Could not save config file:\n{e}")
            return False

def show_config_editor():
    """Displays the API Key Editor window."""
    global config_window, OPENAI_API_KEY, DEEPSEEK_API_KEY, DEEPL_API_KEY

    # Prevent multiple editor windows
    if config_window and config_window.winfo_exists():
        config_window.lift()
        config_window.focus_force()
        return
    else:
        config_window = None # Reset if previous window was closed

    load_api_keys() # Load current keys to display

    try:
        parent = window if window and window.winfo_exists() else None
        config_window = tk.Toplevel(parent) # Create a new top-level window
    except Exception as e_toplevel:
        traceback.print_exc()
        messagebox.showerror("Error", f"Could not create config editor window:\n{e_toplevel}")
        return

    config_window.title("API Key Editor")
    config_window.geometry("550x220")
    config_window.configure(bg="#f0f0f0")
    config_window.resizable(False, False)

    def on_config_close():
        """Closes the config editor window."""
        global config_window
        if config_window and config_window.winfo_exists():
            config_window.destroy()
        config_window = None
    config_window.protocol("WM_DELETE_WINDOW", on_config_close)

    # Attempt to center the window relative to the main window
    try:
        if parent:
            parent.update_idletasks()
            px, py = parent.winfo_x(), parent.winfo_y()
            pw, ph = parent.winfo_width(), parent.winfo_height()
            w, h = 550, 220
            x = px + (pw // 2) - (w // 2)
            y = py + (ph // 2) - (h // 2)
            config_window.geometry(f'{w}x{h}+{x}+{y}')
    except Exception:
        pass # Ignore centering errors

    # --- GUI Elements for the Editor ---
    main_frame = ttk.Frame(config_window, padding=15)
    main_frame.pack(fill=tk.BOTH, expand=True)

    openai_var = tk.StringVar(value=OPENAI_API_KEY or "")
    deepseek_var = tk.StringVar(value=DEEPSEEK_API_KEY or "")
    deepl_var = tk.StringVar(value=DEEPL_API_KEY or "")

    # Input fields
    ttk.Label(main_frame, text="OpenAI Key:", width=15, anchor='w').grid(row=0, column=0, padx=(0,5), pady=5, sticky='w')
    ttk.Entry(main_frame, textvariable=openai_var, width=55).grid(row=0, column=1, padx=5, pady=5, sticky='ew')
    ttk.Label(main_frame, text="DeepSeek Key:", width=15, anchor='w').grid(row=1, column=0, padx=(0,5), pady=5, sticky='w')
    ttk.Entry(main_frame, textvariable=deepseek_var, width=55).grid(row=1, column=1, padx=5, pady=5, sticky='ew')
    ttk.Label(main_frame, text="DeepL Key:", width=15, anchor='w').grid(row=2, column=0, padx=(0,5), pady=5, sticky='w')
    ttk.Entry(main_frame, textvariable=deepl_var, width=55).grid(row=2, column=1, padx=5, pady=5, sticky='ew')
    main_frame.columnconfigure(1, weight=1) # Make entry fields expand

    # --- Save Action ---
    def save_keys_from_editor():
        """Saves keys entered in the editor and reloads."""
        new_openai = openai_var.get().strip()
        new_deepseek = deepseek_var.get().strip()
        new_deepl = deepl_var.get().strip()
        if save_api_keys_to_xml(new_openai, new_deepseek, new_deepl):
            reload_config_and_clients() # Reload everything after saving
            on_config_close() # Close editor on success
        else:
            messagebox.showwarning("Save Failed", "Could not save keys. Please check console output.", parent=config_window)

    # --- Buttons ---
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=3, column=0, columnspan=2, pady=(20, 0), sticky='e')
    save_button = ttk.Button(button_frame, text="Save & Reload Keys", command=save_keys_from_editor, style="Success.TButton")
    save_button.pack(side=tk.RIGHT, padx=(5,0))
    cancel_button = ttk.Button(button_frame, text="Cancel", command=on_config_close, style="Secondary.TButton")
    cancel_button.pack(side=tk.RIGHT)

    # Make the window appear on top initially
    config_window.lift()
    config_window.attributes('-topmost', True)
    config_window.after(100, lambda: config_window.attributes('-topmost', False)) # Release topmost after a short delay
    config_window.focus_force()


# --- History Management ---
def sort_history_data():
    """Sorts the global history_data list by timestamp (newest first)."""
    global history_data
    try:
        # Attempt to parse timestamp and sort
        history_data.sort(key=lambda x: datetime.strptime(x.get("time", "1970-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S"), reverse=True)
    except ValueError as e:
        # Handle potential parsing errors if timestamps are malformed
        print(f"Error sorting history data due to invalid timestamp format: {e}")
        # Optionally show a warning, but don't stop the application
        # messagebox.showwarning("History Sort Warning", f"Could not sort some history entries due to invalid date format: {e}")
    except Exception as e:
        print(f"Unexpected error sorting history data: {e}")
        # messagebox.showerror("History Sort Error", f"Failed to sort history: {e}")

def save_history_to_xml(history_data_to_save):
    """Saves the provided history data list to the HISTORY_FILE (XML format)."""
    # Use a copy to avoid modifying the global list directly during iteration if needed later
    # history_copy = list(history_data_to_save) # Not strictly needed here
    try:
        # Basic directory and permission checks
        history_dir = os.path.dirname(HISTORY_FILE) or "."
        os.makedirs(history_dir, exist_ok=True) # Ensure directory exists
        if os.path.exists(HISTORY_FILE) and not os.access(HISTORY_FILE, os.W_OK):
            raise PermissionError(f"No write permission for {HISTORY_FILE}")

        # Build XML structure
        root = ET.Element("history")
        for entry in history_data_to_save:
            item = ET.SubElement(root, "item")
            for key, value in entry.items():
                child = ET.SubElement(item, key)
                # Ensure values are strings for XML
                child.text = str(value) if value is not None else ""

        # Write prettified XML to file
        xml_bytes = prettify_xml(root)
        with open(HISTORY_FILE, "wb") as f:
            # Write XML declaration explicitly
            # f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n') # prettify_xml handles this
            f.write(xml_bytes)

        # Sanity check if file was created (optional)
        # if not os.path.exists(HISTORY_FILE):
        #     raise FileNotFoundError(f"XML file {HISTORY_FILE} was not created after write.")

    except PermissionError as e:
        print(f"Permission error saving history: {e}")
        messagebox.showerror("History Save Error", f"Permission error writing to '{HISTORY_FILE}':\n{e}")
    except Exception as e:
        print(f"Error saving history: {e}")
        traceback.print_exc()
        messagebox.showerror("History Save Error", f"Failed to save history to '{HISTORY_FILE}':\n{e}")

def load_history_from_path(filepath):
    """Loads history entries from a specified XML file path."""
    loaded_entries = []
    if not filepath:
        return False, "No file path provided.", []
    if not os.path.exists(filepath):
        return False, f"File not found:\n{os.path.basename(filepath)}", []

    try:
        with history_lock: # Use lock for file access
            tree = ET.parse(filepath)
            root = tree.getroot()
            if root.tag != 'history':
                errmsg = f"Invalid root tag '{root.tag}' in file '{os.path.basename(filepath)}'. Expected 'history'."
                return False, errmsg, []

            required_fields = ['time', 'provider', 'original', 'translated', 'rephrased', 'target_language']
            for entry_elem in root.findall('item'):
                entry = {}
                valid_entry = True
                for field in required_fields:
                    child = entry_elem.find(field)
                    entry[field] = child.text if child is not None and child.text is not None else ""
                    # Basic validation (e.g., check if time exists)
                    if field == 'time' and not entry[field]:
                         valid_entry = False; break # Skip entries without a time
                if valid_entry:
                    loaded_entries.append(entry)

        return True, f"Read {len(loaded_entries)} valid entries from\n{os.path.basename(filepath)}", loaded_entries

    except ET.ParseError as e:
        errmsg = f"Error parsing XML file '{os.path.basename(filepath)}': {e}"
        return False, errmsg, []
    except Exception as e:
        errmsg = f"Unexpected error loading history from '{os.path.basename(filepath)}': {e}"
        traceback.print_exc()
        return False, f"An unexpected error occurred:\n{e}", []

def load_history_from_xml():
    """Loads history from the default HISTORY_FILE into the global history_data."""
    global history_data
    history_data = [] # Clear existing in-memory history first
    if not os.path.exists(HISTORY_FILE):
        return # Nothing to load

    try:
        with history_lock: # Use lock for file access
            tree = ET.parse(HISTORY_FILE)
            root = tree.getroot()
            if root.tag != 'history':
                 print(f"Warning: Root tag in {HISTORY_FILE} is not 'history'. Skipping load.")
                 return

            required_fields = ['time', 'provider', 'original', 'translated', 'rephrased', 'target_language']
            for item in root.findall("item"):
                entry = {}
                valid_entry = True
                for field in required_fields:
                    child = item.find(field)
                    entry[field] = child.text if child is not None and child.text is not None else ""
                    if field == 'time' and not entry[field]:
                        valid_entry = False; break # Skip entries without time
                if valid_entry:
                    history_data.append(entry)

        sort_history_data() # Sort loaded data

    except ET.ParseError as e:
        print(f"Error parsing history file {HISTORY_FILE}: {e}")
        messagebox.showerror("History Load Error", f"Failed to parse history file:\n{e}\nHistory might be corrupted.")
    except Exception as e:
        print(f"Error loading history: {e}")
        traceback.print_exc()
        messagebox.showerror("History Load Error", f"Failed to load history from '{HISTORY_FILE}':\n{e}")

def prompt_and_load_history():
    """Prompts the user to select an XML file and merges its content into the current history."""
    global history_data, history_window

    initial_dir = os.path.dirname(os.path.abspath(HISTORY_FILE)) if os.path.exists(HISTORY_FILE) else os.getcwd()
    filepath = filedialog.askopenfilename(
        title="Select History XML File to Load/Merge",
        filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
        initialdir=initial_dir )
    if not filepath: return # User cancelled

    success, message, loaded_entries = load_history_from_path(filepath)

    if success:
        if not loaded_entries:
            messagebox.showinfo("Load History", f"No valid history entries found in\n{os.path.basename(filepath)}")
            return

        # Merge loaded entries with existing data, avoiding duplicates based on timestamp
        existing_times = {entry.get('time') for entry in history_data if entry.get('time')}
        unique_new_entries = [entry for entry in loaded_entries if entry.get('time') not in existing_times]

        if not unique_new_entries:
             messagebox.showinfo("History Merge", f"All {len(loaded_entries)} entries from\n{os.path.basename(filepath)}\nalready exist in the current history.")
             return

        history_data.extend(unique_new_entries)
        sort_history_data()
        save_history_to_xml(history_data) # Save the merged history

        # Update the history window if it's open
        if history_window and history_window.winfo_exists():
            update_history_window_content()

        messagebox.showinfo("History Merged", f"{len(unique_new_entries)} unique entries loaded and merged from\n{os.path.basename(filepath)}\n\nTotal entries now: {len(history_data)}")
    else:
        # Show error message from load_history_from_path
        messagebox.showerror("Load History Error", f"Could not load history:\n\n{message}")

def backup_and_clear_history():
    """Backs up the current history file and clears the history."""
    global history_data, history_window

    backup_filename = ""
    backup_made = False
    if os.path.exists(HISTORY_FILE):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"translation_history_{timestamp}.xml"
        with history_lock:
            try:
                shutil.move(HISTORY_FILE, backup_filename)
                backup_made = True
            except Exception as e:
                messagebox.showerror("History Backup Error", f"Could not backup '{HISTORY_FILE}' to '{backup_filename}':\n{e}")
                traceback.print_exc()
                return # Stop if backup failed

    # Clear in-memory data and save empty file
    history_data.clear()
    save_history_to_xml([]) # Save empty history

    # Update the history window if it's open
    if history_window and history_window.winfo_exists():
        update_history_window_content()

    if backup_made:
        messagebox.showinfo("History Backup & Clear", f"History backed up to:\n{backup_filename}\n\nHistory cleared.")
    else:
        messagebox.showinfo("History Cleared", f"No existing history file found to back up.\nIn-memory history cleared.")


# --- Core Logic: Processing API Results and Updating GUI ---
def update_result(text_passed, full_result, is_rephrase=False, is_reverse=False):
    """
    Processes the result from the API, updates the GUI text boxes,
    and saves an entry to the history.

    Args:
        text_passed (str): The text that was the input for the API call.
                           - Forward Translation: The original source text.
                           - Reverse Translation: The text from the target box (e.g., Turkish).
                           - Rephrase: The original source text (last_selected_text).
        full_result (str | dict): The raw result from the ask_ai function.
                                  String for LLMs or __ERROR__, Dict for DeepL success.
        is_rephrase (bool): True if this is the result of a 'Rephrase Again' action (LLM only).
        is_reverse (bool): True if this is the result of a 'Translate Back' action.
    """
    global last_selected_text, last_translation, history_data
    global original_textbox, translated_textbox, rephrased_textbox
    global last_history_source_language, last_history_target_language # Set before calling API
    provider = api_provider_var.get() if api_provider_var else "N/A"

    try:
        # --- Handle API Errors First ---
        if isinstance(full_result, str) and full_result.startswith("__ERROR__::"):
            error_msg = full_result.replace("__ERROR__::", "")
            messagebox.showerror("API Error", error_msg)
            # Do not proceed further, but update button states
            update_button_states(action_was_tr_to_en=is_reverse)
            return

        # --- Prepare Variables for GUI and History ---
        gui_original = ""       # Text for the top box
        gui_translated = ""     # Text for the middle box
        gui_rephrased = ""      # Text for the bottom box
        history_original = ""   # Original text for history record
        history_translated = "" # Translated text for history record
        history_rephrased = ""  # Rephrased text for history record

        # --- Process Result Based on Action Type ---

        # A. Reverse Translation Result
        if is_reverse:
            history_original = text_passed      # The text that was translated back (e.g., Turkish)
            history_translated = str(full_result) # The result of back-translation (e.g., English)
            history_rephrased = "[N/A for reverse translation]"
            # Update GUI: Show result in Source box, original in Target box
            gui_original = history_translated
            gui_translated = history_original
            gui_rephrased = "" # Clear rephrase box

        # B. Rephrase Again Result (LLM Only)
        elif is_rephrase:
            # This block is now only called by LLM rephrase actions
            processed_rephrase_result = str(full_result)
            # Clean extra newlines from DeepSeek if needed (though called by LLM now)
            if provider == "DeepSeek":
                 processed_rephrase_result = re.sub(r'\n\s*\n', '\n', processed_rephrase_result.strip())

            history_original = text_passed      # Original source text (last_selected_text)
            history_translated = last_translation # Keep the previous translation
            history_rephrased = processed_rephrase_result # The new rephrased text
            # Update GUI: Only update the rephrase box
            gui_original = history_original
            gui_translated = history_translated
            gui_rephrased = history_rephrased

        # C. Forward Translation Result (DeepL or LLM)
        else:
            history_original = text_passed      # The original source text submitted

            # C1. DeepL Forward Translation (result is dict)
            if provider == "DeepL" and isinstance(full_result, dict):
                 history_translated = full_result.get('translated', '[Translation Error]')
                 # Get rephrase result generated automatically by run_translate_rephrase
                 history_rephrased = full_result.get('rephrased', '[Rephrasing Error]')
                 last_translation = history_translated

            # C2. LLM Forward Translation (result is string)
            elif provider != "DeepL" and isinstance(full_result, str):
                 # Parse the "Translation: ... Rephrased: ..." format
                 lines = full_result.strip().split("\n")
                 tr_keyword = f"{last_history_target_language} Translation:"
                 rep_keyword = f"{last_history_source_language} Rephrased:"
                 found_tr, found_rep, cur_section = False, False, None
                 trans_parts, rep_parts = [], []
                 for line in lines:
                     l = line.strip()
                     if not l: continue # Skip empty lines
                     if l.startswith(tr_keyword) and not found_tr: found_tr=True; cur_section="trans"; trans_parts.append(l[len(tr_keyword):].strip())
                     elif l.startswith(rep_keyword) and found_tr and not found_rep: found_rep=True; cur_section="rep"; rep_parts.append(l[len(rep_keyword):].strip())
                     elif cur_section == "trans": trans_parts.append(l)
                     elif cur_section == "rep": rep_parts.append(l)
                 history_translated = "\n".join(trans_parts).strip() if found_tr else "[Translation not found in LLM response]"
                 history_rephrased = "\n".join(rep_parts).strip() if found_rep else "[Rephrasing not found in LLM response]"
                 last_translation = history_translated

            # C3. Unexpected Format for Forward Translation
            else:
                 history_translated = "[Error: Unexpected API result format]"
                 history_rephrased = "[Error: Unexpected API result format]"

            # Assign GUI variables for forward translation
            gui_original = history_original
            gui_translated = history_translated
            gui_rephrased = history_rephrased

            # Update last_selected_text ONLY on successful forward translation
            if history_translated and not history_translated.startswith("["):
                 last_selected_text = history_original
            #else: # Keep the previous last_selected_text if translation failed


        # --- Save to History ---
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_history_entry = {
            "time": ts,
            "original": history_original,
            "translated": history_translated,
            "rephrased": history_rephrased,
            "provider": provider, # Consider adding '(Rephrase)' tag here if is_rephrase? No, handled in rephrase_again.
            "target_language": f"{last_history_source_language} -> {last_history_target_language}"
        }
        history_data.append(new_history_entry)
        sort_history_data()
        try:
            save_history_to_xml(history_data)
        except Exception as e_hist:
            # Avoid showing messagebox here if save_history_to_xml already does
            print(f"Error saving history via update_result: {e_hist}")


        # --- Update GUI Text Boxes ---
        def update_textbox(box, text):
             """Helper function to update a text box safely."""
             if box and box.winfo_exists():
                  try:
                       box.config(state='normal'); box.delete("1.0", tk.END)
                       if text: box.insert(tk.END, text)
                       box.config(state='normal')
                  except tk.TclError: pass # Ignore if widget is destroyed
                  except Exception as e_update: print(f"Unexpected error updating textbox: {e_update}"); traceback.print_exc()

        update_textbox(original_textbox, gui_original)
        update_textbox(translated_textbox, gui_translated)
        update_textbox(rephrased_textbox, gui_rephrased)


        # --- Update Button States ---
        update_button_states(action_was_tr_to_en=is_reverse)

    except Exception as e:
        # Catch unexpected errors during result processing
        print(f"--- FATAL ERROR in update_result ---")
        traceback.print_exc()
        messagebox.showerror("Update Result Fatal Error", f"An critical error occurred processing the result: {str(e)}")
        # Attempt to update buttons even after error
        update_button_states(action_was_tr_to_en=is_reverse)


# --- Button Action Functions ---
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# Gerekli importlar ve global değişkenler yukarıda tanımlı olmalı
import time
import traceback
import deepl
import threading
from tkinter import messagebox
# window, original_textbox, api_provider_var, source_language_var, target_language_var
# deepl_translator, last_history_source_language, last_history_target_language
# get_deepl_source_code, get_deepl_target_code, full_prompt, ask_ai, update_result

def run_translate_rephrase():
    """
    Handles the 'Translate & Rephrase' button click.
    Initiates forward translation (Source -> Target).
    If using DeepL, also performs automatic rephrasing via Target -> Source translation.
    If using LLM, the API is prompted to do both translation and rephrasing.
    """
    global original_textbox, last_history_source_language, last_history_target_language, deepl_translator, window, api_provider_var, source_language_var, target_language_var

    if not original_textbox: return
    text_to_process = original_textbox.get("1.0", tk.END).strip()
    if not text_to_process:
        messagebox.showwarning("Input Missing", "Please enter text in the Source Text area.")
        return

    # Get current settings
    provider = api_provider_var.get() if api_provider_var else "DeepL"
    src_lang_display = source_language_var.get() if source_language_var else "English"
    tgt_lang_display = target_language_var.get() if target_language_var else "Turkish"

    # Set language direction for history (used by update_result)
    last_history_source_language = src_lang_display
    last_history_target_language = tgt_lang_display

    # --- Background API Call ---
    def api_call():
        """Performs the API call in a separate thread."""
        final_result_for_update = "__ERROR__::API Call Failed in thread." # Default error

        try:
            # A. DeepL: Two-step process (Translate, then Back-Translate for Rephrase)
            if provider == "DeepL":
                if not deepl_translator: raise ConnectionError("DeepL translator not initialised.")

                # Language codes for Step 1 (Source -> Target)
                step1_source_lang_code = get_deepl_source_code(src_lang_display) # e.g., TR
                step1_target_lang_code = get_deepl_target_code(tgt_lang_display) # e.g., EN-GB

                # Step 1: Forward Translation
                step1_result = deepl_translator.translate_text(
                    text_to_process, source_lang=step1_source_lang_code, target_lang=step1_target_lang_code )
                translated_text = step1_result.text # e.g., English text

                if not translated_text:
                    final_result_for_update = { "translated": "[Translation Failed or Empty]", "rephrased": "[Rephrasing skipped]" }
                else:
                    # Step 2: Back-Translation (Target -> Source) for Rephrase
                    rephrased_text = "[Rephrasing Failed]" # Default if step 2 fails
                    try:
                        # === DEĞİŞİKLİK BURADA: Step 2 için doğru dil kodları ===
                        # Source for step 2 is the language of translated_text (Target language of step 1)
                        # We need the *base* code for DeepL source_lang if it's English
                        step2_source_lang_code = get_deepl_source_code(tgt_lang_display) # e.g., EN
                        # Target for step 2 is the original source language
                        step2_target_lang_code = get_deepl_target_code(src_lang_display) # e.g., TR
                        # === DEĞİŞİKLİK SONU ===

                        step2_result = deepl_translator.translate_text(
                            translated_text,            # Text from Step 1 (e.g., English)
                            source_lang=step2_source_lang_code, # Correct source (e.g., EN)
                            target_lang=step2_target_lang_code  # Correct target (e.g., TR)
                        )
                        rephrased_text = step2_result.text or "[Rephrasing resulted in empty text]"

                    except deepl.DeepLException as e_rephrase:
                         # Log the specific error but show a cleaner message in GUI
                         print(f"DeepL Rephrase (Step 2) Error: {e_rephrase}")
                         rephrased_text = f"[Rephrasing Error: {type(e_rephrase).__name__}]" # Show only exception type
                    except Exception as e_rephrase_other:
                         print(f"Unexpected Step 2 Error: {e_rephrase_other}"); traceback.print_exc()
                         rephrased_text = f"[Unexpected Rephrasing Error]"

                    # Package results
                    final_result_for_update = { "translated": translated_text, "rephrased": rephrased_text }

            # B. LLM: Single prompt for Translate & Rephrase
            else: # OpenAI or DeepSeek
                prompt_ = full_prompt(text_to_process, src_lang_display, tgt_lang_display)
                final_result_for_update = ask_ai(prompt=prompt_)

        # --- Exception Handling for API Call ---
        except deepl.DeepLException as e_deepl: final_result_for_update = f"__ERROR__::DeepL API Error: {e_deepl}"
        except ConnectionError as e_conn: final_result_for_update = f"__ERROR__::{e_conn}"
        except Exception as e:
             error_message = f"Error during Translate & Rephrase API call: {type(e).__name__}: {e}"
             print(error_message); traceback.print_exc()
             final_result_for_update = f"__ERROR__::{error_message}"
        finally:
            # --- Send Result to Main Thread for GUI Update ---
            if window and window.winfo_exists():
                window.after_idle(update_result, text_to_process, final_result_for_update, False, False)

    # Start the background thread
    threading.Thread(target=api_call, daemon=True).start()
def rephrase_again():
    """
    Handles the 'Rephrase Again' button click.
    Rephrases the 'last_selected_text' based on the current API provider.
    - For LLMs (OpenAI/DeepSeek): Uses a specific rephrase prompt asking for 5 alternatives.
    - For DeepL: Uses a multi-language double-translation technique (EN->X->EN)
                 with several intermediate languages (TR, FR, RU, IT, ES)
                 and lists all successful, unique results. Saves to history.
    """
    global last_selected_text, api_provider_var, window, source_language_var, rephrased_textbox, deepl_translator
    global history_data, last_translation, last_history_source_language, last_history_target_language # For history saving

    # --- Initial Checks and Logging ---
    current_provider = api_provider_var.get() if api_provider_var else "N/A"
    if not last_selected_text:
        messagebox.showwarning("No Text Found", "Cannot rephrase because the source text is missing.\nPlease perform a translation first.")
        return

    source_language_display_name = source_language_var.get() if source_language_var else "English"

    # --- Helper function to update the rephrase text box safely ---
    def update_rephrase_box(text_to_display):
        """Safely updates the rephrased_textbox from a thread."""
        if not (rephrased_textbox and rephrased_textbox.winfo_exists()): return
        try:
            is_error = isinstance(text_to_display, str) and text_to_display.startswith("__ERROR__::")
            display_text = text_to_display
            if is_error:
                error_msg_only = text_to_display.replace("__ERROR__::", "")
                messagebox.showerror("Rephrase Error", f"Could not rephrase:\n\n{error_msg_only}")
                display_text = "[Rephrasing failed, see error message]"
            rephrased_textbox.config(state='normal'); rephrased_textbox.delete("1.0", tk.END)
            rephrased_textbox.insert(tk.END, display_text); rephrased_textbox.config(state='normal')
        except tk.TclError: pass # Ignore if widget destroyed during update
        except Exception as e_update: print(f"Unexpected error in update_rephrase_box: {e_update}"); traceback.print_exc()
        finally: update_button_states(action_was_tr_to_en=False) # Update buttons after action

    # --- Main Logic: DeepL vs LLM ---

    # A. DeepL: Multi-Language Double Translation
    if current_provider == "DeepL":
        if not deepl_translator:
             messagebox.showerror("API Error", "DeepL API Key not configured or translator not initialised."); return

        intermediate_codes = ["TR", "FR", "RU", "IT", "ES"] # Languages for variation
        results = [] # Stores results from threads
        threads = []
        results_lock = threading.Lock() # Thread-safe access to results list

        try:
            source_lang_code = get_deepl_source_code(source_language_display_name) # e.g., 'EN'
            target_lang_code = get_deepl_target_code(source_language_display_name) # e.g., 'EN-GB'
        except Exception as e_lang:
             messagebox.showerror("Language Error", f"Could not get DeepL language codes: {e_lang}"); return

        # --- Thread function for double translation via one intermediate language ---
        def perform_double_translation(intermediate_code, src_code, tgt_code, text, translator, results_list, lock):
            rephrased_output = None; error_output = None
            try:
                # Step 1: Source -> Intermediate
                step1_result = translator.translate_text(text, source_lang=src_code, target_lang=intermediate_code)
                intermediate_text = step1_result.text
                if not intermediate_text: raise ValueError(f"Intermediate ({intermediate_code}) empty")
                # Step 2: Intermediate -> Target (Original Source Language)
                step2_result = translator.translate_text(intermediate_text, source_lang=intermediate_code, target_lang=tgt_code)
                rephrased_output = step2_result.text
                if not rephrased_output: raise ValueError(f"Final ({tgt_code}) empty via {intermediate_code}")
            # Catch specific and general errors
            except deepl.DeepLException as e: error_output = f"DeepL API Error via {intermediate_code}: {e}"
            except ValueError as e: error_output = f"ValueError via {intermediate_code}: {e}"
            except Exception as e: error_output = f"Unexpected error via {intermediate_code}: {type(e).__name__}: {e}"; traceback.print_exc() # Log unexpected
            # Append result or error to shared list safely
            with lock:
                results_list.append(rephrased_output if not error_output else f"__ERROR__::{error_output}")

        # --- Create and start threads ---
        for code in intermediate_codes:
            thread = threading.Thread(target=perform_double_translation, args=(code, source_lang_code, target_lang_code, last_selected_text, deepl_translator, results, results_lock), daemon=True, name=f"DeepL-{code}")
            threads.append(thread); thread.start()

        # --- Function to process results after threads complete ---
        def process_deepl_results():
            # Wait for all threads (with timeout)
            for i, thread in enumerate(threads):
                 thread.join(timeout=60.0)
                 if thread.is_alive(): print(f"Warning: Thread {thread.name} timed out.")

            # Collect all successful, non-error results
            valid_rephrases = []
            with results_lock: # Access shared list safely
                for res in results:
                    if isinstance(res, str) and not res.startswith("__ERROR__::"):
                         res_strip = res.strip()
                         if res_strip: # Ensure it's not empty after stripping
                            valid_rephrases.append(res_strip)
                    # else: log error if needed

            # Format the output list
            if valid_rephrases:
                display_limit = 5
                formatted_output = "\n".join(f"{i+1}. {phrase}" for i, phrase in enumerate(valid_rephrases[:display_limit]))
                if len(valid_rephrases) > display_limit: formatted_output += f"\n... ({len(valid_rephrases)} results found)"
            else:
                formatted_output = "[No valid rephrased alternatives found using multi-language translation.]"

            # --- Save DeepL Rephrase to History ---
            try:
                history_original_text = last_selected_text
                history_translated_text = last_translation or "[Previous translation missing]"
                history_rephrased_text = formatted_output # Save the formatted list
                translation_direction = f"{last_history_source_language} -> {last_history_target_language}" or "Unknown Direction"
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_history_entry = { "time": ts, "original": history_original_text, "translated": history_translated_text, "rephrased": history_rephrased_text, "provider": f"{current_provider} (Rephrase)", "target_language": translation_direction }
                history_data.append(new_history_entry); sort_history_data(); save_history_to_xml(history_data)
            except Exception as e_hist: print(f"!!! Error saving DeepL rephrase to history: {e_hist}"); traceback.print_exc()

            # --- Update GUI ---
            if window and window.winfo_exists():
                 window.after(0, update_rephrase_box, formatted_output)

        # Start a separate thread to wait for and process results, keeping GUI responsive
        threading.Thread(target=process_deepl_results, daemon=True).start()


    # B. LLM (OpenAI / DeepSeek) for Normal Rephrase
    else:
        try: src_lang_for_llm = supported_languages.get(current_provider, {}).get(source_language_display_name, source_language_display_name)
        except Exception as e_lang: messagebox.showerror("Language Error", f"Could not get LLM language name: {e_lang}"); return
        prompt_ = rephrase_prompt(last_selected_text, src_lang_for_llm)

        # --- Background thread for LLM API call ---
        def llm_api_call():
            raw_llm_result = None # Initialise
            try:
                raw_llm_result = ask_ai(prompt=prompt_) # Returns string or error string
                if raw_llm_result is None: raw_llm_result = "__ERROR__::LLM API call returned None."
                # Let update_result handle history and GUI update
                if window and window.winfo_exists():
                    window.after_idle(update_result, last_selected_text, raw_llm_result, True, False)
            except Exception as e:
                error_message = f"Unexpected error during LLM rephrase thread: {type(e).__name__}: {e}"; print(error_message); traceback.print_exc()
                error_result = f"__ERROR__::{error_message}"
                # Send error to update_result for history logging and GUI update
                if window and window.winfo_exists(): window.after_idle(update_result, last_selected_text, error_result, True, False)
                update_button_states(action_was_tr_to_en=False) # Update buttons on error too

        threading.Thread(target=llm_api_call, daemon=True).start()


def translate_to_source():
    """
    Handles the 'Translate Back to Source' button click.
    Takes text from the target box, translates it to the selected source language,
    updates the source text box, and saves to history.
    """
    global translated_textbox, last_history_source_language, last_history_target_language, window, api_provider_var, source_language_var, target_language_var

    if not translated_textbox: return
    text_to_translate_back = translated_textbox.get("1.0", tk.END).strip()
    if not text_to_translate_back:
        messagebox.showwarning("Input Missing", "Please enter text in the 'Translated Text (Target)' area to translate back.")
        return

    # Get current settings
    provider = api_provider_var.get() if api_provider_var else "DeepL"
    # Target of this operation is the original Source Language
    final_target_lang_display = source_language_var.get() if source_language_var else "English"
    # Source of this operation is the current Target Language
    current_text_lang_display = target_language_var.get() if target_language_var else "Turkish"

    # Set history direction for this specific action
    last_history_source_language = current_text_lang_display # e.g., Turkish
    last_history_target_language = final_target_lang_display # e.g., English

    # --- Background API Call ---
    def api_call():
        final_result_for_update = "__ERROR__::Back-translation failed." # Default error

        try:
            # A. DeepL Back-Translation
            if provider == "DeepL":
                source_lang_code = get_deepl_source_code(current_text_lang_display) # e.g., TR
                target_lang_code = get_deepl_target_code(final_target_lang_display) # e.g., EN-GB
                deepl_dict_result = ask_ai( None, text_to_translate_back, target_lang_code, source_lang_code )
                if isinstance(deepl_dict_result, dict) and "translated" in deepl_dict_result:
                    final_result_for_update = deepl_dict_result["translated"]
                elif isinstance(deepl_dict_result, str) and deepl_dict_result.startswith("__ERROR__::"):
                    final_result_for_update = deepl_dict_result
                else: final_result_for_update = "__ERROR__::DeepL Error: Unexpected back-translation format."

            # B. LLM Back-Translation
            else: # OpenAI or DeepSeek
                prompt_ = translate_to_source_prompt(text_to_translate_back, final_target_lang_display)
                raw_llm_result = ask_ai(prompt=prompt_)
                if isinstance(raw_llm_result, str):
                    if raw_llm_result.startswith("__ERROR__::"): final_result_for_update = raw_llm_result
                    else:
                        # Clean potential LLM formatting prefixes
                        cleaned_res = raw_llm_result.strip().replace('"', '')
                        kw = f"{final_target_lang_display} Translation:"
                        final_result_for_update = cleaned_res[len(kw):].strip() if cleaned_res.lower().startswith(kw.lower()) else cleaned_res
                else: final_result_for_update = f"__ERROR__::LLM Error: Unexpected back-translation type ({type(raw_llm_result)})."

        except Exception as e:
            error_message = f"Error during back-translation thread: {type(e).__name__}: {e}"; print(error_message); traceback.print_exc()
            final_result_for_update = f"__ERROR__::{error_message}"
        finally:
             # Send result to main thread
             if window and window.winfo_exists():
                 # text_to_translate_back is Original for history (e.g., Turkish)
                 # final_result_for_update is Translated for history (e.g., English)
                 window.after_idle(update_result, text_to_translate_back, final_result_for_update, False, True)

    # Start the background thread
    threading.Thread(target=api_call, daemon=True).start()


# --- GUI Creation ---
def show_gui(make_visible=False):
    """Creates or shows the main application window."""
    # Ensure all required globals are accessible
    global window, original_textbox, translated_textbox, rephrased_textbox
    global rephrase_button, translate_button, reverse_translate_button, history_button
    global style_dropdown, api_dropdown, target_language_dropdown, source_language_dropdown
    global font_size, style_var, api_provider_var, target_language_var, source_language_var
    # Add necessary imports if not already global
    global deepl_translator, DEEPSEEK_API_KEY, openai_client, supported_languages, style_options
    global update_gui_after_reload, update_button_states, run_translate_rephrase, translate_to_source, rephrase_again, show_history

    # If window exists, just show it
    if window is not None:
        try:
            if make_visible:
                if window.state() == 'withdrawn': window.deiconify()
                window.lift(); window.focus_force()
            return
        except tk.TclError: window = None
        except Exception as e: print(f"Error reactivating window: {e}"); window = None

    # Create the main window if it doesn't exist
    if window is None:
        window = tk.Tk()
        window.title("Translator & Rephraser Pro")
        window.geometry("900x750")
        window.minsize(850, 600)
        window.configure(bg="#f0f0f0")

        # --- Configure Styles ---
        style = ttk.Style(window)
        try:
            theme = 'clam' # Use clam as a reasonable default
            style.theme_use(theme)
        except tk.TclError:
            try: style.theme_use('xpnative') # Fallback for Windows
            except tk.TclError: style.theme_use('default') # Final fallback
        except Exception as e: print(f"Error setting theme: {e}"); style.theme_use('default')
        # Define custom styles (assuming these are correct from previous code)
        style.configure("TFrame", background="#f0f0f0")
        style.configure("Content.TFrame", background="#ffffff", borderwidth=1, relief="solid")
        style.configure("TLabelframe", background="#f0f0f0", borderwidth=0, relief="flat")
        style.configure("TLabelframe.Label", background="#f0f0f0", foreground="#333333", font=('Segoe UI', 10, 'bold'))
        style.configure("TLabel", background="#f0f0f0", foreground="#333333", font=('Segoe UI', 9))
        style.configure("Bold.TLabel", background="#f0f0f0", foreground="#333333", font=('Segoe UI', 9, 'bold'))
        style.configure("Header.TLabel", background="#f0f0f0", foreground="#333333", font=('Segoe UI', 10, 'bold'))
        style.configure("TButton", padding=(8, 6), relief="flat", font=('Segoe UI', 9), borderwidth=0)
        style.configure("Accent.TButton", foreground="white", background="#007bff", font=('Segoe UI', 9, 'bold'), borderwidth=0)
        style.map("Accent.TButton", foreground=[('disabled', '#ffffff')], background=[('active', '#0056b3'), ('disabled', '#a0c7e4')])
        style.configure("Success.TButton", foreground="white", background="#28a745", font=('Segoe UI', 9, 'bold'), borderwidth=0)
        style.map("Success.TButton", foreground=[('disabled', '#ffffff')], background=[('active', '#218838'), ('disabled', '#a3d9b1')])
        style.configure("Warning.TButton", foreground="#212529", background="#ffc107", font=('Segoe UI', 9, 'bold'), borderwidth=0)
        style.map("Warning.TButton", foreground=[('disabled', '#6c757d')], background=[('active', '#e0a800'), ('disabled', '#ffeeba')])
        style.configure("History.TButton", foreground="#ffffff", background="#6c757d", font=('Segoe UI', 9), borderwidth=0)
        style.map("History.TButton", foreground=[('disabled', '#ffffff')], background=[('active', '#5a6268'), ('disabled', '#e2e6ea')])
        style.configure("Small.TButton", padding=1, font=('Segoe UI', 7), borderwidth=0)
        style.configure("Control.Small.TButton", foreground="#333333", background="#e0e0e0", font=('Segoe UI', 7, 'bold'))
        style.map("Control.Small.TButton", background=[('active', '#c8c8c8'), ('disabled', '#f5f5f5')])
        style.configure("Clear.Small.TButton", foreground="white", background="#dc3545", font=('Segoe UI', 7, 'bold'))
        style.map("Clear.Small.TButton", background=[('active', '#c82333'), ('disabled', '#f8d7da')])
        style.configure("TextArea.TFrame", background="white", borderwidth=1, relief="solid")

        window.columnconfigure(0, weight=1); window.rowconfigure(1, weight=1)
        def on_close(): window.withdraw();
        window.protocol("WM_DELETE_WINDOW", on_close)

        # --- Top Control Panel ---
        top_controls_frame = ttk.Frame(window, padding=(10, 10, 10, 5)); top_controls_frame.grid(row=0, column=0, sticky="ew")
        top_controls_frame.columnconfigure(4, weight=1) # Spacer

        # API Dropdown
        api_frame = ttk.Frame(top_controls_frame); api_frame.grid(row=0, column=0, padx=(0, 5), sticky="w")
        ttk.Label(api_frame, text="API:", style="Bold.TLabel").pack(side="left", padx=(0, 5))
        available_providers = ["DeepL"] if deepl_translator else []
        if DEEPSEEK_API_KEY: available_providers.append("DeepSeek")
        if openai_client: available_providers.append("OpenAI")
        if not available_providers: available_providers.append("No APIs Configured")
        default_provider = available_providers[0] if available_providers[0] != "No APIs Configured" else "No APIs Configured"
        if api_provider_var is None: api_provider_var = tk.StringVar(value=default_provider)
        api_dropdown = ttk.Combobox(api_frame, textvariable=api_provider_var, values=available_providers, state="readonly", width=12, font=('Segoe UI', 9)); api_dropdown.pack(side="left")
        if default_provider == "No APIs Configured": api_dropdown.config(state="disabled")
        api_provider_var.trace_add("write", lambda *a: update_gui_after_reload())

        # Source Language Dropdown
        source_lang_frame = ttk.Frame(top_controls_frame); source_lang_frame.grid(row=0, column=1, padx=(5,5), sticky="w")
        ttk.Label(source_lang_frame, text="Source:", style="Bold.TLabel").pack(side="left", padx=(0, 5))
        if source_language_var is None: source_language_var = tk.StringVar(value="English")
        lang_source_vals = list(supported_languages.get(api_provider_var.get(), {}).keys()) or ["English"]
        source_language_dropdown = ttk.Combobox(source_lang_frame, textvariable=source_language_var, values=lang_source_vals, state="readonly", width=15, font=('Segoe UI', 9)); source_language_dropdown.pack(side="left")
        source_language_var.trace_add("write", lambda *a: update_button_states())

        # Target Language Dropdown
        target_lang_frame = ttk.Frame(top_controls_frame); target_lang_frame.grid(row=0, column=2, padx=(5,5), sticky="w")
        ttk.Label(target_lang_frame, text="Target:", style="Bold.TLabel").pack(side="left", padx=(0, 5))
        if target_language_var is None: target_language_var = tk.StringVar(value="Turkish")
        lang_target_vals = list(supported_languages.get(api_provider_var.get(), {}).keys()) or ["Turkish"]
        target_language_dropdown = ttk.Combobox(target_lang_frame, textvariable=target_language_var, values=lang_target_vals, state="readonly", width=15, font=('Segoe UI', 9)); target_language_dropdown.pack(side="left")
        target_language_var.trace_add("write", lambda *a: update_button_states())

        # Style Dropdown
        style_frame = ttk.Frame(top_controls_frame); style_frame.grid(row=0, column=3, padx=(5, 15), sticky="w")
        ttk.Label(style_frame, text="Style:", style="Bold.TLabel").pack(side="left", padx=(0, 5))
        if style_var is None: style_var = tk.StringVar(value="Simple English")
        style_dropdown = ttk.Combobox(style_frame, textvariable=style_var, values=list(style_options.keys()), state="disabled", width=18, font=('Segoe UI', 9)); style_dropdown.pack(side="left")

        # History Button
        history_button = ttk.Button(top_controls_frame, text="🕒 History", command=show_history, style="History.TButton", width=10); history_button.grid(row=0, column=5, sticky="e")

        # --- Main Content Area ---
        content_area = ttk.Frame(window, padding=(10, 5, 10, 10)); content_area.grid(row=1, column=0, sticky="nsew"); content_area.columnconfigure(0, weight=1)
        content_area.rowconfigure(0, weight=1); content_area.rowconfigure(1, weight=0); content_area.rowconfigure(2, weight=1); content_area.rowconfigure(3, weight=0); content_area.rowconfigure(4, weight=1); content_area.rowconfigure(5, weight=0)

        # --- Text Area Creator Helper ---
        def create_ui_text_area(parent_frame, grid_row, label_text, label_icon):
            # --- Frame ve Widget Oluşturma (Aynı) ---
            area_frame = ttk.Frame(parent_frame, padding=(0, 5, 0, 5))
            area_frame.grid(row=grid_row, column=0, sticky="nsew", pady=(0, 5))
            area_frame.columnconfigure(0, weight=1); area_frame.rowconfigure(1, weight=1)
            # Header (Label + Controls)
            header_controls_frame = ttk.Frame(area_frame)
            header_controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 3))
            header_controls_frame.columnconfigure(0, weight=1)
            ttk.Label(header_controls_frame, text=f"{label_icon} {label_text}", style="Header.TLabel").grid(row=0, column=0, sticky="w", padx=(2,0))
            controls_frame = ttk.Frame(header_controls_frame); controls_frame.grid(row=0, column=1, sticky="e")
            # Text Box Container
            text_container = ttk.Frame(area_frame, style="TextArea.TFrame")
            text_container.grid(row=1, column=0, sticky="nsew")
            text_container.columnconfigure(0, weight=1); text_container.rowconfigure(0, weight=1)
            # Text Widget and Scrollbar
            text_box = tk.Text(text_container, wrap=tk.WORD, font=('Segoe UI', font_size), height=7, padx=8, pady=5, bd=0, highlightthickness=0, relief="flat", undo=True, selectbackground="#007bff", selectforeground="white")
            scrollbar = ttk.Scrollbar(text_container, orient="vertical", command=text_box.yview)
            text_box.configure(yscrollcommand=scrollbar.set)
            text_box.grid(row=0, column=0, sticky="nsew"); scrollbar.grid(row=0, column=1, sticky="ns")

            # <<< Bağlam Menüsünü Ekle (Bu zaten vardı) >>>
            add_text_widget_context_menu(text_box)

            # === DEĞİŞİKLİK: Helper Fonksiyonları Düzelt ===
            # Clear fonksiyonu
            def clear_action():
                """Clears the content of the text box."""
                try:
                    if text_box and text_box.winfo_exists():
                        text_box.config(state='normal')
                        text_box.delete("1.0", tk.END)
                        text_box.edit_reset() # Clear undo/redo stack
                except tk.TclError:
                    pass # Widget might be destroyed

            # Font değiştirme fonksiyonu
            def change_font_action(delta):
                """Changes the font size of the text box."""
                try:
                    if not (text_box and text_box.winfo_exists()): return
                    # Get current font properties
                    current_font = tkFont.Font(font=text_box.cget("font"))
                    actual_font_info = current_font.actual()
                    current_size = abs(actual_font_info.get('size', font_size)) # Use abs for negative sizes on some systems
                    # Calculate new size with limits
                    new_size = max(8, min(20, current_size + delta))
                    # Apply new font settings
                    text_box.config(font=(
                        actual_font_info.get('family', 'Segoe UI'), # Keep family
                        new_size,
                        actual_font_info.get('weight', 'normal'), # Keep weight
                        actual_font_info.get('slant', 'roman')     # Keep slant
                    ))
                except tk.TclError:
                    pass # Widget might be destroyed
                except Exception as e:
                    # Log other potential errors (e.g., issues getting font info)
                    print(f"Error changing font size: {e}")
            # === DEĞİŞİKLİK SONU ===

            # Control Buttons (+, -, C) - Komutları düzeltilmiş fonksiyonlara bağla
            ttk.Button(controls_frame, text="C", width=2, command=clear_action, style="Clear.Small.TButton", takefocus=False).pack(side="right", padx=(2, 0))
            ttk.Button(controls_frame, text="-", width=2, command=lambda: change_font_action(-1), style="Control.Small.TButton", takefocus=False).pack(side="right", padx=(2, 0))
            ttk.Button(controls_frame, text="+", width=2, command=lambda: change_font_action(1), style="Control.Small.TButton", takefocus=False).pack(side="right", padx=(5, 0))

            return text_box
        # --- Text Alanı Oluşturma Fonksiyonu Sonu ---

        # --- Create Text Areas and Buttons ---
        button_width=35
        original_textbox = create_ui_text_area(content_area, 0, "Source Text", "📋")
        act_frame1 = ttk.Frame(content_area); act_frame1.grid(row=1, column=0, sticky="e", pady=(5,10))
        translate_button = ttk.Button(act_frame1, text="Translate...", style="Success.TButton", command=run_translate_rephrase, width=button_width); translate_button.pack()
        translated_textbox = create_ui_text_area(content_area, 2, "Translated Text (Target)", "🌐")
        act_frame2 = ttk.Frame(content_area); act_frame2.grid(row=3, column=0, sticky="e", pady=(5,10))
        reverse_translate_button = ttk.Button(act_frame2, text="Translate Back...", style="Accent.TButton", command=translate_to_source, width=button_width); reverse_translate_button.pack()
        rephrased_textbox = create_ui_text_area(content_area, 4, "Rephrased Text (Source)", "🇬🇧")
        act_frame3 = ttk.Frame(content_area); act_frame3.grid(row=5, column=0, sticky="e", pady=(5,0))
        rephrase_button = ttk.Button(act_frame3, text="Rephrase Again", style="Warning.TButton", command=rephrase_again, width=button_width); rephrase_button.pack()

        # --- Initialise GUI State ---
        update_gui_after_reload() # Ensure correct initial state based on loaded keys
        window.withdraw() # Start hidden
        if make_visible: window.deiconify() # Show if requested


# --- Button/Widget State Updates ---
def update_button_states(action_was_tr_to_en=False):
    """Updates the state and text of buttons and dropdowns based on current selections and state."""
    global rephrase_button, translate_button, reverse_translate_button, api_provider_var, style_var, last_selected_text, style_dropdown, api_dropdown, target_language_var, target_language_dropdown, source_language_var, source_language_dropdown

    def do_update():
        """Performs the actual update logic. Called via after_idle."""
        # Check if all required widgets exist before proceeding
        required_widgets = [ window, rephrase_button, translate_button, reverse_translate_button, api_provider_var, api_dropdown, target_language_var, target_language_dropdown, source_language_var, source_language_dropdown, style_dropdown, style_var ]
        if not all(widget and getattr(widget, 'winfo_exists', lambda: True)() for widget in required_widgets):
            return # Skip update if GUI elements are not ready or destroyed

        try:
            provider = api_provider_var.get()
            target_lang = target_language_var.get() or "Target"
            source_lang = source_language_var.get() or "Source"
            is_api_usable = (provider != "No APIs Configured")
            # Rephrase is possible if API is usable and source text exists (and not immediately after a back-translation)
            has_text_for_rephrase = bool(last_selected_text and not action_was_tr_to_en)

            # --- Update Translate Button ---
            state_trans = 'normal' if is_api_usable else 'disabled'
            if provider == "DeepL": text_trans = f"Translate ({source_lang} -> {target_lang})"
            else: text_trans = f"Translate & Rephrase ({source_lang} -> {target_lang})"
            if not is_api_usable: text_trans = "Translate & Rephrase" # Default text when disabled
            translate_button.configure(text=text_trans, state=state_trans)

            # --- Update Reverse Translate Button ---
            state_rev = 'normal' if is_api_usable else 'disabled'
            text_rev = f"Translate Back ({target_lang} -> {source_lang})"
            reverse_translate_button.configure(text=text_rev, state=state_rev)

            # --- Update Rephrase Button ---
            # Active if API is usable AND there's text to rephrase
            state_rephrase = 'normal' if (is_api_usable and has_text_for_rephrase) else 'disabled'
            rephrase_button.configure(state=state_rephrase)

            # --- Update Style Dropdown ---
            # Active if API is usable (even for DeepL, though it has no effect)
            state_style = 'readonly' if is_api_usable else 'disabled'
            style_dropdown.configure(state=state_style)

            # --- Update Language Dropdowns ---
            state_lang = 'readonly' if is_api_usable else 'disabled'
            target_language_dropdown.configure(state=state_lang)
            source_language_dropdown.configure(state=state_lang)

        except tk.TclError: pass # Ignore errors if widgets are destroyed during update
        except Exception as e: print(f"Error updating button/widget states: {e}"); traceback.print_exc()

    # Schedule the update using after_idle for safety
    if window and window.winfo_exists(): window.after_idle(do_update)


# --- History Window ---
def update_history_window_content():
    """Helper function to refresh the content of the history window if open."""
    global history_window, history_data
    if not (history_window and history_window.winfo_exists()): return

    try:
        # Find the Text widget within the history window structure
        hist_text_widget = None
        for widget in history_window.winfo_children():
            if isinstance(widget, ttk.Frame): # Assuming content is in a Frame
                for sub_widget in widget.winfo_children():
                    if isinstance(sub_widget, tk.Text):
                        hist_text_widget = sub_widget; break
            if hist_text_widget: break

        if hist_text_widget:
            hist_text_widget.config(state='normal')
            hist_text_widget.delete("1.0", tk.END)
            if not history_data:
                hist_text_widget.insert(tk.END, "History is empty.\n")
            else:
                history_content = []
                for item in history_data: # Assumes history_data is sorted
                    entry = (f"=== 🕒 {item.get('time', 'N/A')} (API: {item.get('provider', 'N/A')}, Lang: {item.get('target_language', 'N/A')}) ===\n"
                             f"📋 Original:\n{item.get('original', '')}\n\n"
                             f"🌐 Translation:\n{item.get('translated', '')}\n\n"
                             f"🇬🇧 Rephrased:\n{item.get('rephrased', '[N/A]')}\n"
                             + "-"*60 + "\n\n")
                    history_content.append(entry)
                hist_text_widget.insert(tk.END, "".join(history_content))
            hist_text_widget.config(state='disabled')
            hist_text_widget.yview_moveto(0.0) # Scroll to top
    except Exception as e:
        print(f"Could not update open history window: {e}")

def show_history():
    """Displays the translation history window."""
    global history_window, history_data # Ensure history_data is accessible
    # Ensure helper is accessible if defined elsewhere, or define it here
    # global update_history_window_content

    # If window exists, bring it to front
    if history_window and history_window.winfo_exists():
        history_window.lift(); history_window.focus_force(); return

    parent = window if window and window.winfo_exists() else None
    history_window = tk.Toplevel(parent)
    history_window.title("Translation History")
    history_window.geometry("880x620")
    history_window.configure(bg="#f0f0f0")
    history_window.columnconfigure(0, weight=1); history_window.rowconfigure(0, weight=1)

    # Frame for content and scrollbar
    hist_frame = ttk.Frame(history_window, padding=5, style="Content.TFrame")
    hist_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    hist_frame.rowconfigure(0, weight=1); hist_frame.columnconfigure(0, weight=1)

    # Text widget to display history
    hist_textbox = tk.Text(hist_frame, wrap=tk.WORD, font=("Consolas", 11), bd=0, highlightthickness=0, relief="flat", padx=5, pady=5, background="white", state='disabled')
    hist_scrollbar = ttk.Scrollbar(hist_frame, orient="vertical", command=hist_textbox.yview)
    hist_textbox.configure(yscrollcommand=hist_scrollbar.set)
    hist_textbox.grid(row=0, column=0, sticky="nsew"); hist_scrollbar.grid(row=0, column=1, sticky="ns")

    # Populate the text box using the defined helper function
    update_history_window_content() # Call the helper to fill the content

    # <<< Add context menu to history text box >>>
    add_text_widget_context_menu(hist_textbox)

    # Note: The previous manual context menu code for history is removed
    # as add_text_widget_context_menu handles Cut/Copy/Paste/Select All.
# --- Window Management and Hotkey ---
def safe_deiconify():
    """Safely deiconifies, raises, and focuses the main window."""
    if window and window.winfo_exists():
        try:
            if window.state() == 'withdrawn': window.deiconify()
            window.attributes('-topmost', True); window.lift(); window.focus_force()
            window.after(100, lambda: window.attributes('-topmost', False)) # Release topmost
        except tk.TclError: pass # Ignore if window destroyed during operation
        except Exception as e: print(f"Error during deiconify/focus: {e}"); traceback.print_exc()

def process_clipboard_text():
    """Processes text from the clipboard: pastes it into the source box and triggers translation."""
    global hotkey_processing, window, original_textbox
    if hotkey_processing: return # Prevent concurrent processing
    hotkey_processing = True
    try:
        text = pyperclip.paste()
        if text and text.strip():
            if not (window and window.winfo_exists() and original_textbox):
                messagebox.showerror("Error", "Main window or text box not ready.")
                hotkey_processing = False; return

            # Show window first
            safe_deiconify()

            # Schedule the text update and translation trigger slightly later
            def scheduled_processing(clipboard_text):
                global hotkey_processing
                try:
                    if original_textbox and original_textbox.winfo_exists():
                        original_textbox.config(state='normal')
                        original_textbox.delete("1.0", tk.END)
                        original_textbox.insert(tk.END, clipboard_text)
                        # Call run_translate_rephrase to start the process
                        run_translate_rephrase()
                except Exception as e_sched:
                     print(f"Error in scheduled clipboard processing: {e_sched}")
                     traceback.print_exc()
                finally:
                    # Ensure flag is reset even if errors occur
                    hotkey_processing = False # Reset flag here

            # Use after_idle to ensure window is visible before processing
            window.after_idle(lambda t=text: scheduled_processing(t))

        else:
            messagebox.showwarning("Clipboard Empty", "No text found in clipboard.")
            hotkey_processing = False # Reset flag if clipboard is empty
    except Exception as e:
        print(f"Error processing clipboard: {e}"); traceback.print_exc()
        messagebox.showerror("Clipboard/Processing Error", f"Could not process clipboard: {e}")
        hotkey_processing = False # Reset flag on error

def listen_ctrl_c_c():
    """Listens for double Ctrl+C presses to trigger clipboard processing."""
    global hotkey_processing
    double_press_threshold = 0.4 # Max time between presses (seconds)
    last_ctrl_c_time = 0

    def on_ctrl_c():
        nonlocal last_ctrl_c_time
        current_time = time.time()
        if (current_time - last_ctrl_c_time) < double_press_threshold:
            if not hotkey_processing:
                # Schedule the action on the main GUI thread
                if window: window.after(0, process_clipboard_text)
            last_ctrl_c_time = 0 # Reset timer after double press
        else:
            last_ctrl_c_time = current_time # Record time of first press

    try:
        # Register the hotkey (trigger on key down)
        keyboard.add_hotkey('ctrl+c', on_ctrl_c, trigger_on_release=False)
    except ImportError:
        print("Warning: 'keyboard' library not found or requires root/admin privileges for global hotkeys.")
        messagebox.showwarning("Hotkey Warning", "Could not register global hotkey (Ctrl+C+C).\nThis might require Administrator privileges or the 'keyboard' library.")
    except Exception as e:
        print(f"Could not register hotkey: {e}"); traceback.print_exc()
        messagebox.showwarning("Hotkey Error", f"Could not register hotkey:\n{e}")


# --- System Tray Icon Setup ---
def exit_app(icon=None, item=None):
    """Stops the tray icon and exits the application cleanly."""
    global tray_icon, window
    print("Exiting application...")
    if tray_icon and tray_icon.visible:
        try: tray_icon.stop()
        except Exception as e: print(f"Error stopping tray icon: {e}")
    if window:
        try: window.quit(); window.destroy()
        except Exception: pass # Ignore errors during destroy
    # Use os._exit(0) for a more forceful exit if needed, especially if threads hang
    sys.exit(0) # Standard exit

def setup_tray_icon_thread():
    """Sets up and runs the system tray icon in a separate thread."""
    global tray_icon
    try:
        # Load icon image (provide a default if not found)
        icon_path = "deep_translator.png" # Ensure this icon exists
        try:
             image = Image.open(icon_path)
             # Ensure RGB format for compatibility
             if image.mode != 'RGB': image = image.convert('RGB')
        except FileNotFoundError:
             print(f"Warning: Icon file '{icon_path}' not found. Using default.")
             image = Image.new("RGB", (64, 64), color=(30, 130, 90)) # Default green icon
        except Exception as e_img:
             print(f"Error loading icon: {e_img}. Using default.")
             image = Image.new("RGB", (64, 64), color=(30, 130, 90))

    except Exception as e_pil: # Catch errors if PIL is missing
        print(f"Error initialising PIL Image: {e_pil}. Tray icon unavailable.")
        return # Cannot proceed without PIL

    # Helper to schedule actions on the main GUI thread
    def schedule_action(action_func):
        if window and window.winfo_exists(): window.after(0, action_func)

    # Define tray menu items
    menu = Menu(
        MenuItem("Translate Clipboard (Ctrl+C+C)", lambda: schedule_action(process_clipboard_text), default=True),
        MenuItem("Show Window", lambda: schedule_action(safe_deiconify)),
        MenuItem("Show History", lambda: schedule_action(show_history)),
        Menu.SEPARATOR,
        MenuItem("Load History File...", lambda: schedule_action(prompt_and_load_history)),
        MenuItem("Backup & Clear History", lambda: schedule_action(backup_and_clear_history)),
        Menu.SEPARATOR,
        MenuItem("Edit API Keys...", lambda: schedule_action(show_config_editor)),
        MenuItem("Reload Config & Keys", lambda: schedule_action(reload_config_and_clients)),
        Menu.SEPARATOR,
        MenuItem("Exit", exit_app) )

    # Create and run the icon
    tray_icon = Icon("TranslatorPro", image, "Translator & Rephraser Pro", menu=menu)
    # Start hotkey listener in a separate thread
    threading.Thread(target=listen_ctrl_c_c, daemon=True).start()
    try:
        tray_icon.run() # Blocks until exit_app is called
    except Exception as e_tray:
        print(f"Error running tray icon: {e_tray}")
        traceback.print_exc()


# --- Main Execution Block ---
if __name__ == "__main__":
    # Handle Ctrl+C in console for graceful exit
    signal.signal(signal.SIGINT, lambda sig, frame: exit_app())

    # Initial setup
    load_api_keys()
    reinitialize_clients()

    # Warn if no APIs are configured
    api_configured = bool(openai_client or DEEPSEEK_API_KEY or deepl_translator)
    if not api_configured:
        messagebox.showwarning("API Keys Missing",
                               "No valid API keys found in config.xml.\n"
                               "Please add your API keys via 'Edit API Keys...' "
                               "in the tray menu to enable functionality.",
                               icon='warning')

    load_history_from_xml() # Load history on startup
    show_gui(make_visible=False) # Create GUI but keep it hidden initially

    # Start the system tray icon in its own thread
    tray_thread = threading.Thread(target=setup_tray_icon_thread, daemon=True)
    tray_thread.start()

    # Start the Tkinter main loop (blocks)
    if window:
        try:
            window.mainloop()
        except Exception as e_main:
            print("Error in main loop:"); traceback.print_exc()
            exit_app() # Ensure exit if mainloop fails
    else:
        print("Error: Main window could not be created.")
        sys.exit(1)

    # This part might not be reached if exit_app is called directly
    # from the tray menu or signal handler, which is fine.
    print("Main loop finished.")
    exit_app()