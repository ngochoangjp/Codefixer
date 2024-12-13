import gradio as gr
import ollama
import os
import subprocess
import time
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Ollama Functions ---
def get_environment_context():
    context = os.getenv("ENVIRONMENT_CONTEXT")
    if not context:
        return "Not specified"
    return context

def prepare_prompt(code, error_message="", environment_context=""):
    prompt = "You are a coding assistant tasked with fixing and improving code.\n"
    if environment_context:
        prompt += f"Environment: {environment_context}\n"
    if error_message:
        prompt += f"Error: {error_message}\n"
    prompt += f"Code:\n```\n{code}\n```\n"
    prompt += "Provide corrected code and explanations."
    return prompt

def fix_code(code, error_message="", model="mistral"):
    environment_context = get_environment_context()
    prompt = prepare_prompt(code, error_message, environment_context)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response['response']
    except Exception as e:
        return f"Error during code analysis: {e}"

# --- Terminal Functions ---
def run_code_in_terminal(code, filename="temp_code.py"):
    with open(filename, "w") as f:
        f.write(code)

    with subprocess.Popen(
        ["python", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    ) as process:
        for line in process.stdout:
            yield line
        for line in process.stderr:
            yield line

# --- Gradio Interface ---
def reset_terminal_content():
    return ""

def handle_code_submission(code, error_message, model):
    suggestions = fix_code(code, error_message, model)
    return suggestions, ""

def handle_code_execution(code):
    for output_line in run_code_in_terminal(code):
        yield output_line

def start_ollama(model_name: str):
    try:
        # Check if model is already running
        result = ollama.list()
        print("Running models:", result)  # Debug print
        
        # Start the model
        subprocess.run(['ollama', 'run', model_name], check=True)
        return f"Model '{model_name}' started successfully."
    except subprocess.CalledProcessError as e:
        return f"Failed to start model '{model_name}': {e}"
    except Exception as e:
        return f"Error starting model: {str(e)}"

def stop_ollama():
    try:
        subprocess.run(['ollama', 'stop'], check=True)
        return "Ollama stopped successfully."
    except subprocess.CalledProcessError as e:
        return f"Failed to stop Ollama: {e}"
    
def reset_ollama():
    try:
        subprocess.run(['ollama', 'rm', '-f', '*'], check=True)
        return "Ollama resetted successfully."
    except subprocess.CalledProcessError as e:
        return f"Failed to stop Ollama: {e}"

def get_ollama_models():
    try:
        result = ollama.list()
        print("Available models:", result)  # Debug print
        if isinstance(result, dict) and 'models' in result:
            return [model.get('name', '') for model in result['models'] if model.get('name')]
        return ["mistral"]  # Default fallback model
    except Exception as e:
        print(f"Error getting models: {e}")  # Debug print
        return ["mistral"]  # Default fallback model

# --- Gradio Interface ---
with gr.Blocks(title="Ollama Code Fixer") as demo:
    ollama_models = get_ollama_models()

    with gr.Row():
        with gr.Column():
            code_input = gr.Code(label="Code to Fix", language="python")
            error_input = gr.Textbox(label="Error Message (optional)")

            model_dropdown = gr.Dropdown(
                label="Select Ollama Model", choices=ollama_models, value=ollama_models[0] if ollama_models else None
            )

            with gr.Accordion("Ollama Controls", open=False):
                ollama_start_button = gr.Button("Start Ollama Model")
                ollama_stop_button = gr.Button("Stop Ollama")
                ollama_reset_button = gr.Button("Reset Ollama")
                ollama_status = gr.Textbox(label="Status", value="Ready")

            submit_button = gr.Button("Get Suggestions")
            
        with gr.Column():
            ai_suggestions = gr.Code(label="AI Suggestions", interactive=True)
            fix_button = gr.Button("Apply Fix")

        with gr.Column():
            terminal_output = gr.Textbox(label="Terminal Output", lines=10)
            run_button = gr.Button("Run Code")
            reset_button = gr.Button("Reset Terminal")

    # --- Event Handlers ---
    submit_button.click(
        handle_code_submission,
        inputs=[code_input, error_input, model_dropdown],
        outputs=[ai_suggestions, terminal_output],
    )

    fix_button.click(lambda x: x, inputs=[ai_suggestions], outputs=[code_input])

    run_button.click(
        handle_code_execution,
        inputs=[code_input],
        outputs=[terminal_output],
    )

    reset_button.click(
        reset_terminal_content,
        outputs=[terminal_output]
    )

    ollama_start_button.click(
        start_ollama,
        inputs=[model_dropdown],
        outputs=[ollama_status]
    )
    ollama_stop_button.click(
        stop_ollama,
        inputs=None,
        outputs=[ollama_status]
    )
    ollama_reset_button.click(
        reset_ollama,
        inputs=None,
        outputs=[ollama_status]
    )

    demo.queue()
    demo.launch(max_threads=20)