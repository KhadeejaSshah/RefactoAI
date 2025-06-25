import os
import ast
import torch
import signal
import sys
import gc
import logging
import difflib
import re
import tempfile
from io import StringIO
from typing import Dict, List, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from radon.complexity import cc_visit
from radon.metrics import mi_visit, mi_rank
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.staticfiles import StaticFiles

class Settings(BaseSettings):
    frontend_url: str = "http://localhost:3000"
    class Config:
        env_file = ".env"

settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

deepseek_model = None
deepseek_tokenizer = None
model_phi = None
tokenizer_phi = None

def cleanup():
    global deepseek_model, deepseek_tokenizer, model_phi, tokenizer_phi
    logger.info("Cleanup: Releasing GPU memory...")
    try:
        if deepseek_model is not None:
            del deepseek_model
            deepseek_model = None
            logger.debug("Deleted deepseek_model")
        if deepseek_tokenizer is not None:
            del deepseek_tokenizer
            deepseek_tokenizer = None
            logger.debug("Deleted deepseek_tokenizer")
        if model_phi is not None:
            del model_phi
            model_phi = None
            logger.debug("Deleted model_phi")
        if tokenizer_phi is not None:
            del tokenizer_phi
            tokenizer_phi = None
            logger.debug("Deleted tokenizer_phi")
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Cleanup: GPU memory has been successfully released.")
    except Exception as e:
        logger.error(f"Cleanup Error: {e}")

def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}. Initiating cleanup...")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_deepseek_model(device):
    global deepseek_model, deepseek_tokenizer
    logger.info("Loading DeepSeek Model for Code Generation...")
    try:
        deepseek_model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=True
        )
        deepseek_tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            use_auth_token=True
        )
        deepseek_tokenizer.pad_token = deepseek_tokenizer.eos_token  
        logger.info("DeepSeek Model Loaded Successfully")
    except Exception as e:
        logger.error(f"Failed to load DeepSeek model: {e}")
        raise e

def load_phi_model(device):
    global model_phi, tokenizer_phi
    logger.info("Loading Microsoft Phi-4-mini-instruct Model for User Story Generation...")
    try:
        model_name = "microsoft/Phi-4-mini-instruct"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=True
        )
        logger.info("Microsoft Phi-4-mini-instruct Model Loaded Successfully")
        model_phi = model
        tokenizer_phi = tokenizer
    except Exception as e:
        logger.error(f"Failed to load Microsoft Phi-4-mini-instruct model: {e}")
        raise e

@app.on_event("startup")
def startup_event():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Backend starting on device: {device}")
    try:
        load_deepseek_model(device)
        load_phi_model(device)
        logger.info("All models loaded successfully.")
    except Exception as e:
        logger.critical(f"Error during startup: {e}")
        sys.exit(1)

@app.on_event("shutdown")
def shutdown_event():
    cleanup()

def check_syntax_content(source: str) -> (bool, str):
    try:
        ast.parse(source)
        logger.debug("AST syntax check passed.")
        return True, ""
    except SyntaxError as e:
        logger.error(f"AST SyntaxError: {e}")
        return False, f"SyntaxError: {e.msg} at line {e.lineno}, column {e.offset}"
    except Exception as e:
        logger.error(f"Unexpected Error during syntax check: {e}")
        return False, f"Unexpected error: {e}"

def check_pep8_content(source: str) -> (bool, str):
    pep8_errors = StringIO()
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp_file:
            tmp_file.write(source)
            tmp_file_path = tmp_file.name
        import pycodestyle
        style = pycodestyle.StyleGuide(quiet=False, stdout=pep8_errors)
        report = style.check_files([tmp_file_path])
        if report.total_errors == 0:
            logger.info("[PEP8 Check] Code conforms to PEP8.")
            return True, ""
        else:
            logger.warning(f"[PEP8 Check] Found {report.total_errors} PEP8 violation(s).")
            return False, pep8_errors.getvalue()
    except Exception as e:
        logger.error(f"Error during PEP8 checking: {e}")
        return False, f"Error during PEP8 checking: {e}"
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
                logger.debug(f"Deleted temporary PEP8 file: {tmp_file_path}")
            except Exception as e:
                logger.error(f"Error deleting temporary PEP8 file: {e}")

def check_code_quality_content(source: str) -> dict:
    try:
        complexity = cc_visit(source)
        complexity_info = []
        if not complexity:
            complexity_info.append(" - No functions or classes to analyze for complexity.")
        for block in complexity:
            complexity_info.append(f" - {block.name} (line {block.lineno}): Cyclomatic Complexity = {block.complexity}")
        mi_score = mi_visit(source, False)
        mi_category = mi_rank(mi_score)
        maintainability_info = f" - Maintainability Index: {mi_score:.2f} ({mi_category})"
        return {
            "complexity": "\n".join(complexity_info),
            "maintainability": maintainability_info
        }
    except ImportError:
        logger.error("radon is not installed.")
        return {"complexity": "radon is not installed.", "maintainability": "radon is not installed."}
    except Exception as e:
        logger.error(f"Error during code quality analysis: {e}")
        return {"complexity": "Error during analysis.", "maintainability": "Error during analysis."}

def extract_user_stories(text: str) -> List[str]:
    split_pattern = r'\n\d+\.\s|\n- |\nReq \d+\.\s|\n{2,}'  # Splits on common patterns and double newlines.
    raw_user_stories = re.split(split_pattern, text)
    req_pattern = r'Req\.?\s*\d+\.\s*(.*)'
    req_matches = re.findall(req_pattern, text)
    unique_user_stories = list(dict.fromkeys(raw_user_stories + req_matches))  # Remove exact duplicates
    return [req.strip() for req in unique_user_stories if req.strip()]

def has_only_imports(source: str) -> bool:
    """
    Returns True if the provided Python source code is empty or contains only
    import statements (and optionally docstrings), otherwise False.
    """
    try:
        tree = ast.parse(source)
        if not tree.body:
            return True
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            # Allow docstrings (the first statement can be a string literal)
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Str) or (hasattr(node.value, "value") and isinstance(node.value.value, str)):
                    continue
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking imports: {e}")
        return False

def analyze_requirements_with_phi(requirements: str, device: str, phi_max_length: int, phi_beam_size: int, phi_temperature: float) -> str:
    if not all([model_phi, tokenizer_phi]):
        logger.error("Microsoft Phi-4-mini-instruct model or tokenizer is not loaded.")
        raise HTTPException(status_code=500, detail="Microsoft Phi-4-mini-instruct model not loaded.")
    prompt = (
        "You are an expert software analyst. "
        "Transform the following requirements into a list of concise user stories. Do NOT include any extra instructions or text.\n"
        "Each user story must be a single concise line that starts with 'as a user, i want...' and ends with '...so that...', with exactly one blank line separating each story.\n\n"
        "If the user provides requirements which are not related to coding or technology or python or refactoring , then tell the user that its not related and dont produce any user stories.\n\n"
        f"Requirements:\n{requirements}\n\n"
        "User Stories:\n"
    )
    try:
        input_ids = tokenizer_phi(prompt, return_tensors="pt").input_ids.to(device)
        outputs = model_phi.generate(
            input_ids,
            max_length=phi_max_length,
            num_beams=phi_beam_size,
            no_repeat_ngram_size=2,
            early_stopping=True,
            temperature=phi_temperature,
            top_p=0.9,
            eos_token_id=tokenizer_phi.eos_token_id,
        )
        gen_text = tokenizer_phi.decode(outputs[0], skip_special_tokens=True).strip()
        logger.debug(f"[Raw Phi Output] {gen_text}")
        lower_gen_text = gen_text.lower()
        user_stories_start_idx = lower_gen_text.find("user stories:")
        if user_stories_start_idx != -1:
            gen_text = gen_text[user_stories_start_idx + len("user stories:"):].strip()
        user_stories_list = extract_user_stories(gen_text)
        final_user_stories = "\n\n".join(user_stories_list)
        return final_user_stories
    except Exception as e:
        logger.error(f"Error during user story generation with Phi: {e}")
        raise HTTPException(status_code=500, detail="Error generating user stories with Microsoft Phi-4-mini-instruct.")

def integrate_user_stories_with_deepseek(original_code: str, user_stories: str, analysis_output: str, device: str, deepseek_max_length: int, deepseek_beam_size: int, deepseek_temperature: float) -> str:
    if not all([deepseek_model, deepseek_tokenizer]):
        logger.error("DeepSeek model or tokenizer is not loaded.")
        raise HTTPException(status_code=500, detail="DeepSeek model not loaded.")
    cleaned_user_stories = "\n\n".join(extract_user_stories(user_stories))
    prompt = (
        "You are a python developer. "
        "Incorporate the following user stories into the given Python code. Additionally, refactor and improve the code. "
        "Make the Cyclomatic Complexity and Maintainability Index of the existing code better, and ensure that any PEP8 violations mentioned are corrected.\n\n"
        "Preserve existing functionality and add new features directly. "
        "Return only the new, modified Python code. "
        "No explanations, no disclaimers, no original code repeated.\n\n"
        "The code should be proper, runnable python code without any extraneous markers or symbols.\n\n"
        f"Original Code:\n{original_code}\n\n"
        f"User Stories:\n{cleaned_user_stories}\n\n"
        "Final Python Code:\n"
    )
    logger.info("Modifying script with DeepSeek...")
    try:
        inputs = deepseek_tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        gen_tokens = deepseek_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=deepseek_max_length,
            do_sample=True if deepseek_temperature > 0 else False,
            temperature=deepseek_temperature if deepseek_temperature > 0 else None,
            num_beams=deepseek_beam_size if deepseek_beam_size > 1 else None,
            pad_token_id=deepseek_tokenizer.eos_token_id
        )
        modified_code = deepseek_tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        if prompt in modified_code:
            modified_code = modified_code.replace(prompt, "").strip()
        cleaned_lines = []
        for line in modified_code.splitlines():
            line_lower = line.lower().strip()
            if line_lower.startswith("original code:") or line_lower.startswith("user stories:"):
                continue
            cleaned_lines.append(line)
        modified_code = "\n".join(cleaned_lines).strip()
        modified_code = re.sub(r'^</think>\s*', '', modified_code)
        modified_code = re.sub(r'^python\s*', '', modified_code)
        modified_code = re.sub(r'$', '', modified_code, flags=re.MULTILINE)
        modified_code = "\n".join(cleaned_lines).strip()
        modified_code = re.sub(r'^</think>\s*', '', modified_code)
        modified_code = re.sub(r'^```python\s*', '', modified_code)
        modified_code = re.sub(r'```$', '', modified_code, flags=re.MULTILINE)
        idx = modified_code.lower().find("import")
        if idx != -1:
            modified_code = modified_code[idx:]
        logger.info("Code modified successfully.")
        return modified_code
    except Exception as e:
        logger.error(f"Error during code modification: {e}")
        return None

def extract_code_lines(code: str) -> List[Tuple[int, str]]:
    """Extracts all lines from the code along with line numbers."""
    return [(i + 1, line.strip()) for i, line in enumerate(code.split("\n")) if line.strip()]

def extract_functions_and_classes_from_code(code: str) -> Dict[str, List[Tuple[int, str, str]]]:
    """Extract function and class definitions from the code."""
    functions = []
    classes = []
    lines = code.split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("def "):
            func_name = line.split("(")[0][4:]
            functions.append((i, func_name, line))
        elif line.startswith("class "):
            class_name = line.split("(")[0][6:]
            classes.append((i, class_name, line))
    return {"functions": functions, "classes": classes}

def get_semantic_similarity(text1: str, text2: str, model) -> float:
    """Calculate semantic similarity between two texts."""
    embeddings = model.encode([text1, text2])
    return (embeddings[0] @ embeddings[1]) / (sum(embeddings[0]**2)**0.5 * sum(embeddings[1]**2)**0.5)

def generate_report(user_stories_str: str, modified_code: str) -> Dict[str, Dict[str, str]]:
    """Generates a report by checking whether each requirement is met in the modified code."""
    report = {}
    user_stories = extract_user_stories(user_stories_str)
    code_info = extract_functions_and_classes_from_code(modified_code)
    all_code_lines = extract_code_lines(modified_code)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for idx, requirement in enumerate(user_stories, 1):
        requirement_id = f"REQ{idx}"
        highest_similarity = 0.0
        best_match = None
        
        # First, check only function/class names
        for line_num, func_name, func_content in code_info['functions']:
            sim_score = get_semantic_similarity(requirement, func_name, model)
            if sim_score > highest_similarity:
                highest_similarity = sim_score
                best_match = f"Function: {func_name} (Line {line_num + 1})"

        for line_num, cls_name, cls_content in code_info['classes']:
            sim_score = get_semantic_similarity(requirement, cls_name, model)
            if sim_score > highest_similarity:
                highest_similarity = sim_score
                best_match = f"Class: {cls_name} (Line {line_num + 1})"

        # Define threshold for function/class match
        threshold = 0.4  # Base threshold
        status = "Met" if highest_similarity >= threshold else "Not Met"
        
        # If not met, scan the entire code line by line
        if status == "Not Met":
            for line_num, line_content in all_code_lines:
                sim_score = get_semantic_similarity(requirement, line_content, model)
                if sim_score > highest_similarity:
                    highest_similarity = sim_score
                    best_match = f"Line {line_num}: {line_content}"
            threshold = max(0.4, highest_similarity - 0.05)
            status = "Met" if highest_similarity >= threshold else "Not Met"
        print("reeeeeqqqqqqqqqq = ", requirement)    
        story = requirement
        report[requirement_id] = {
            "user_story": story,
            "requirement_met": status,
            "matched_elements": best_match if best_match else "N/A"
        }
    
    return report

def generate_diff_structured(original_code: str, modified_code: str) -> list:
    diff_lines = list(difflib.ndiff(original_code.splitlines(), modified_code.splitlines()))
    structured = []
    for line in diff_lines:
        if line.startswith('- ') or line.startswith('? '):
            continue
        elif line.startswith('+ '):
            structured.append({"text": line[2:], "added": True})
        elif line.startswith('  '):
            structured.append({"text": line[2:], "added": False})
    return structured

@app.post("/analyze-and-modify/")
async def analyze_and_modify(
    file: UploadFile = File(...),
    requirements: str = Form(...),
    phi_max_length: int = Form(512),
    phi_beam_size: int = Form(5),
    phi_temperature: float = Form(0.7),
    deepseek_max_length: int = Form(6500),
    deepseek_beam_size: int = Form(1),
    deepseek_temperature: float = Form(0.0)
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Received a request to /analyze-and-modify/")
    try:
        contents = await file.read()
        original_code = contents.decode('utf-8')
        logger.info(f"Processing file: {file.filename}")
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")
    
    # New feature: Check if file is empty or contains only imports (and docstrings)
    if not original_code.strip():
        raise HTTPException(status_code=400, detail="Your file has no code in it (empty file).")
    if has_only_imports(original_code):
        raise HTTPException(status_code=400, detail="Your file has no code in it (only library imports).")
    
    try:
        syntax_ok, syntax_errors = check_syntax_content(original_code)
        if not syntax_ok:
            logger.warning("Syntax errors detected in uploaded code.")
            raise HTTPException(status_code=400, detail=f"Syntax errors detected:\n{syntax_errors}")
        logger.info("Syntax check passed.")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error during syntax checking: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during syntax checking.")
    
    try:
        pep8_ok, pep8_errors = check_pep8_content(original_code)
        if not pep8_ok:
            logger.info("PEP8 violations detected.")
        else:
            logger.info("PEP8 check passed.")
    except Exception as e:
        logger.error(f"Error during PEP8 checking: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during PEP8 checking.")
    
    try:
        code_quality = check_code_quality_content(original_code)
        logger.info("Original code quality analysis completed.")
    except Exception as e:
        logger.error(f"Error during code quality analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during code quality analysis.")
    
    try:
        user_stories = analyze_requirements_with_phi(requirements, device, phi_max_length, phi_beam_size, phi_temperature)
        if not user_stories:
            logger.error("Failed to generate user stories.")
            raise HTTPException(status_code=500, detail="Failed to generate user stories.")
        logger.info("User stories generated successfully.")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error during user story generation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during user story generation.")
    
    analysis_output = ""
    if pep8_errors.strip():
        analysis_output += "PEP8 Issues:\n" + pep8_errors.strip() + "\n\n"
    else:
        analysis_output += "PEP8 Issues: None\n\n"
    analysis_output += "Code Quality Analysis:\n"
    analysis_output += "Complexity:\n" + code_quality.get("complexity", "") + "\n"
    analysis_output += code_quality.get("maintainability", "")
    
    try:
        modified_code = integrate_user_stories_with_deepseek(original_code, user_stories, analysis_output, device, deepseek_max_length, deepseek_beam_size, deepseek_temperature)
        if not modified_code:
            logger.error("Failed to modify the code.")
            raise HTTPException(status_code=500, detail="Failed to modify the code.")
        logger.info("Code modified successfully.")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error during code modification: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during code modification.")
    
    try:
        result_report = generate_report(user_stories, modified_code)
        logger.info("Report generated successfully.")
    except Exception as e:
        logger.error(f"Error during report generation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during report generation.")
    
    try:
        modified_code_quality = check_code_quality_content(modified_code)
        modified_code_quality["pep8_issues"] = pep8_errors.strip() if pep8_errors.strip() else "None"
        logger.info("Modified code quality analysis completed.")
    except Exception as e:
        logger.error(f"Error during modified code quality analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during modified code quality analysis.")
    
    diff = generate_diff_structured(original_code, modified_code)
    logger.info("Successfully processed the request.")
    return {
        "original_code": original_code,
        "syntax": {
            "errors": syntax_errors,
            "status": "ok" if not syntax_errors else "errors found"
        },
        "pep8": {
            "violations": pep8_errors,
            "status": "ok" if not pep8_errors.strip() else "violations found"
        },
        "code_quality": code_quality,
        "user_stories": user_stories,
        "modified_code": modified_code,
        "diff": diff,
        "report": result_report,
        "modified_code_quality": modified_code_quality,
        "generation_parameters": {
            "phi": {
                "max_length": phi_max_length,
                "beam_size": phi_beam_size,
                "temperature": phi_temperature
            },
            "deepseek": {
                "max_length": deepseek_max_length,
                "beam_size": deepseek_beam_size,
                "temperature": deepseek_temperature
            }
        }
    }

@app.get("/health")
def health_check():
    if all([deepseek_model, deepseek_tokenizer, model_phi, tokenizer_phi]):
        return {"status": "ok", "message": "All models loaded."}
    else:
        return {"status": "error", "message": "Models not loaded."}, 500

app.mount("/", StaticFiles(directory="build", html=True), name="build")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
