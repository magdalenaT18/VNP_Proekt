from flask import Flask, request, send_file, render_template, url_for, redirect, jsonify
import os
import fitz  # PyMuPDF
from gliner import GLiNER
from datetime import datetime
from werkzeug.utils import secure_filename
import json
import redis
import re
import time
from apscheduler.schedulers.background import BackgroundScheduler
import cv2
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import easyocr
import logging
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score
import traceback
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['PREVIEW_FOLDER'] = 'static/previews'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['MAX_PREVIEW_PAGES'] = 3

# Anthropic API key configuration
app.config['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', '')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREVIEW_FOLDER'], exist_ok=True)
import anthropic

anthropic_client = anthropic.Anthropic(
    api_key=app.config['ANTHROPIC_API_KEY']
)

#################################
# Anthropic Integration Only
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    app.logger.warning("Anthropic library not installed. Install with: pip install anthropic")

def ask_anthropic(prompt, model="claude-3-haiku-20240307"):
    """Query Anthropic Claude API"""
    if not ANTHROPIC_AVAILABLE:
        return "Anthropic library not installed. Please install with: pip install anthropic"
    
    if not app.config['ANTHROPIC_API_KEY']:
        return "Anthropic API key not configured. Please set ANTHROPIC_API_KEY environment variable."
    
    try:
        client = anthropic.Anthropic(api_key=app.config['ANTHROPIC_API_KEY'])
        
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.1,  # Low temperature for consistent PII detection
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
    except anthropic.AuthenticationError:
        return "Invalid Anthropic API key. Please check your API key configuration."
    except anthropic.RateLimitError:
        return "Anthropic API rate limit exceeded. Please try again later."
    except Exception as e:
        app.logger.error(f"Anthropic error: {str(e)}")
        return f"Error with Anthropic: {str(e)}"



def analyze_pdf_with_anthropic(text):
    """
    Analyze text for PII using Claude with maximum detection sensitivity and chunking for long documents
    """
    
    def chunk_text(text, chunk_size=8000, overlap=500):
        """Split text into overlapping chunks to handle long documents"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at word boundaries
            if end < len(text):
                while end > start and text[end] not in ' \n\t.!?':
                    end -= 1
                if end == start:  # Fallback if no good break point
                    end = min(start + chunk_size, len(text))
            
            chunk = text[start:end]
            chunks.append((chunk, start))
            
            if end >= len(text):
                break
                
            start = end - overlap
        
        return chunks
    
    def ultra_aggressive_prompt(chunk_text, chunk_offset=0):
        """Ultra-aggressive PII detection prompt"""
        return f"""MAXIMUM PII DETECTION - FIND EVERY POSSIBLE ENTITY

You are a PII detection expert. Scan this text fragment and find ALL potential PII. Miss NOTHING.

DETECTION TARGETS (with examples):
• Names: John Smith, Dr. Johnson, Mary, J.D., Mr. Brown, Ms. Wilson, LLC, Corp, Inc
• Emails: ANY text with @ (user@email.com, contact@site.org)  
• Phones: 555-1234, (555)123-4567, 555.123.4567, +1-555-123-4567, 1-800-555-1234
• Addresses: 123 Main St, 456 Oak Ave, PO Box 789, Suite 100, Apt 4B, Room 205
• SSN: 123-45-6789, 123456789, XXX-XX-XXXX
• Credit Cards: 4111-1111-1111-1111, 4111111111111111 (13-19 digits)
• Dates: 1/1/1990, 01-15-1985, Jan 1 1990, DOB: 1990, born 1985, age 35
• IDs: EMP123, ID-456, #789, Account 12345, Policy P123456, License L789
• Organizations: When linked to people (John works at Microsoft, Mary from Apple Inc)

AGGRESSIVE RULES:
1. Scan EVERY word - if it's capitalized and could be a name, include it
2. Find ALL @ symbols and extract full email addresses
3. Detect ALL number sequences that could be phone/ID/SSN/card numbers
4. Include ALL dates, especially potential birth dates
5. Find addresses by looking for: number + street type (St/Ave/Rd/Dr/Ln/Blvd)
6. If uncertain, ALWAYS include it - false positives are better than misses
7. Look for variations: John J., J. Smith, Smith, J., etc.
8. Find organizational names when associated with people

SCAN THIS FRAGMENT:
{chunk_text}

Return ONLY JSON - calculate positions from start of this fragment:
{{"entities": [
    {{"type": "Person", "text": "exact match", "start": 0, "end": 4}}
]}}

Types: Person, Email, Phone, Address, Ssn, Credit_Card, Date_of_birth, ID_Number, Organization

FIND EVERYTHING NOW!"""

    def merge_entities(all_entities):
        """Merge entities from multiple chunks, adjusting positions and removing duplicates"""
        merged = []
        seen_entities = set()
        
        for entities in all_entities:
            for entity in entities:
                # Create a unique identifier for deduplication
                entity_id = (entity['type'], entity['text'].lower(), entity.get('start', 0))
                if entity_id not in seen_entities:
                    merged.append(entity)
                    seen_entities.add(entity_id)
        
        # Sort by start position
        merged.sort(key=lambda x: x.get('start', 0))
        return merged

    try:
        print(f"Processing text of length: {len(text)}")
        
        # For very long texts, use chunking
        if len(text) > 6000:
            print("Text is long, using chunking approach...")
            chunks = chunk_text(text, chunk_size=6000, overlap=300)
            print(f"Split into {len(chunks)} chunks")
            
            all_chunk_entities = []
            
            for i, (chunk, offset) in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)} (offset: {offset})")
                
                chunk_prompt = ultra_aggressive_prompt(chunk, offset)
                
                try:
                    response = anthropic_client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=4096,
                        temperature=0.1,  # Slight randomness for better detection
                        messages=[{"role": "user", "content": chunk_prompt}]
                    )
                    
                    content = response.content[0].text.strip()
                    
                    # Parse JSON
                    chunk_entities = []
                    try:
                        json_data = json.loads(content)
                        if json_data.get("entities"):
                            # Adjust positions to account for chunk offset
                            for entity in json_data["entities"]:
                                entity['start'] += offset
                                entity['end'] += offset
                            chunk_entities = json_data["entities"]
                    except json.JSONDecodeError:
                        # Try regex extraction
                        import re
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            try:
                                json_data = json.loads(json_match.group())
                                if json_data.get("entities"):
                                    for entity in json_data["entities"]:
                                        entity['start'] += offset
                                        entity['end'] += offset
                                    chunk_entities = json_data["entities"]
                            except:
                                pass
                    
                    print(f"  Found {len(chunk_entities)} entities in chunk {i+1}")
                    all_chunk_entities.append(chunk_entities)
                    
                except Exception as e:
                    print(f"Error processing chunk {i+1}: {e}")
                    all_chunk_entities.append([])
            
            # Merge all entities
            all_entities = []
            for chunk_entities in all_chunk_entities:
                all_entities.extend(chunk_entities)
            
            merged_entities = merge_entities([all_entities])
            print(f"Total entities after merging: {len(merged_entities)}")
            
            return {"entities": merged_entities}
        
        else:
            # For shorter texts, use single pass
            print("Using single-pass detection...")
            prompt = ultra_aggressive_prompt(text)
            
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text.strip()
            print(f"Response preview: {content[:300]}...")
            
            # Parse response
            try:
                json_data = json.loads(content)
                entities = json_data.get("entities", [])
                print(f"Found {len(entities)} entities")
                
                # If still low detection, try desperate measures
                if len(entities) < 20:  # Arbitrary threshold for "low"
                    print("Low detection count, trying emergency mode...")
                    
                    emergency_prompt = f"""EMERGENCY: The text below contains many PII entities that you missed.

Go through WORD BY WORD:
1. Every capitalized word might be a name - include it
2. Every @ symbol means an email - extract it  
3. Every sequence of digits might be phone/ID/SSN - include it
4. Every date format might be a birthday - include it
5. Every street address pattern - include it

BE EXTREMELY LIBERAL. Include anything remotely suspicious.

TEXT:
{text}

JSON ONLY: {{"entities": [...]}}"""

                    try:
                        emergency_response = anthropic_client.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=4096,
                            temperature=0.3,
                            messages=[{"role": "user", "content": emergency_prompt}]
                        )
                        
                        emergency_content = emergency_response.content[0].text.strip()
                        emergency_json = json.loads(emergency_content)
                        emergency_entities = emergency_json.get("entities", [])
                        
                        if len(emergency_entities) > len(entities):
                            print(f"Emergency mode found {len(emergency_entities)} entities (better!)")
                            return emergency_json
                    except Exception as e:
                        print(f"Emergency mode failed: {e}")
                
                return json_data
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                # Try regex extraction
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except:
                        pass
                
                return {"entities": [], "error": "Could not parse JSON"}
        
    except Exception as e:
        print(f"Error in analyze_pdf_with_anthropic: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"entities": [], "error": str(e)}

@app.route('/analyze_pdf', methods=['POST'])
def analyze_pdf():
    """Analyze uploaded PDF with both GLiNER and Anthropic for comparison"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    # Save uploaded file
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = secure_filename(f"{timestamp}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Extract text from PDF
        doc = fitz.open(filepath)
        full_text = ""
        page_count = len(doc)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text() or ""
            full_text += f"\n[Page {page_num + 1}]\n{page_text}"
        
        doc.close()

        if not full_text.strip():
            return jsonify({'error': 'No text could be extracted from PDF'}), 400

        print(f"Extracted text length: {len(full_text)} characters")
        print(f"Text preview: {full_text[:200]}...")

        # Analyze with GLiNER
        model = GLiNER.from_pretrained("E3-JSI/gliner-multi-pii-domains-v1")
        pii_labels = ["person", "email", "phone", "address", "ssn", "credit_card", "passport", "date_of_birth"]
        
        gliner_entities, gliner_time, gliner_metadata = measure_gliner_response_time(
            model, full_text, pii_labels, threshold=0.7
        )

        # Filter valid GLiNER entities
        valid_gliner_entities = []
        for ent in gliner_entities:
            entity_text = full_text[ent['start']:ent['end']].strip()
            if is_valid_pii(ent, entity_text, ent['label']):
                valid_gliner_entities.append({
                    'text': entity_text,
                    'type': ent['label'],
                    'confidence': f"{ent['score']:.2f}",
                    'start': ent['start'],
                    'end': ent['end']
                })

        print(f"GLiNER found {len(valid_gliner_entities)} valid entities")

        # Analyze with Anthropic
        anthropic_start_time = time.perf_counter()
        anthropic_result = analyze_pdf_with_anthropic(full_text)
        anthropic_end_time = time.perf_counter()
        anthropic_time = (anthropic_end_time - anthropic_start_time) * 1000

        print(f"Anthropic analysis completed in {anthropic_time:.2f}ms")
        print(f"Anthropic result: {anthropic_result}")

        # Count Anthropic entities
        anthropic_entities = anthropic_result.get('entities', [])
        anthropic_count = len(anthropic_entities)

        # Prepare comparison results
        comparison_result = {
            'file_info': {
                'filename': file.filename,
                'pages': page_count,
                'text_length': len(full_text),
                'timestamp': timestamp
            },
            'gliner_analysis': {
                'entities': valid_gliner_entities,
                'processing_time_ms': round(gliner_time, 2),
                'total_entities': len(valid_gliner_entities),
                'categories': list(set([e['type'] for e in valid_gliner_entities]))
            },
            'anthropic_analysis': {
                'entities': anthropic_entities,
                'processing_time_ms': round(anthropic_time, 2),
                'total_entities': anthropic_count,
                'error': anthropic_result.get('error', None)
            },
            'comparison': {
                'gliner_faster': gliner_time < anthropic_time,
                'time_difference_ms': round(abs(anthropic_time - gliner_time), 2),
                'entity_count_difference': len(valid_gliner_entities) - anthropic_count
            }
        }

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify(comparison_result)

    except Exception as e:
       
        try:
            os.remove(filepath)
        except:
            pass
        
        app.logger.error(f"PDF analysis error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
################################
@app.route("/llm_redact", methods=["POST"])
def llm_redact():
    text = request.form.get("text")
    
    # Enhanced prompt for better PII detection
    prompt = f"""You are a privacy protection assistant. Analyze the following text and identify all personally identifiable information (PII).

For each piece of PII found, provide:
1. The exact text/value
2. The PII type (Name, Email, Phone, SSN, Address, Credit Card, etc.)
3. Location in text (if possible)

PII Types to look for:
- Names (first, last, full names)
- Email addresses
- Phone numbers (any format)
- Social Security Numbers (SSN)
- Credit card numbers
- Addresses (street, city, state, zip)
- Dates of birth
- Driver's license numbers
- Passport numbers
- Medical record numbers
- Bank account numbers
- IP addresses
- Any other sensitive personal information

Text to analyze:
{text}

If no PII is found, state "No personally identifiable information detected."

Format your response clearly with bullet points for each PII item found."""

    # Use Anthropic for PII detection
    result = ask_anthropic(prompt)
    
    # Return JSON for AJAX requests
    if request.headers.get('Content-Type') == 'application/json':
        return jsonify({
            "llm_result": result,
            "provider": "anthropic",
            "timestamp": datetime.now().isoformat()
        })
    
    # Return HTML template for direct form submissions
    return render_template('result.html', 
                         result=result, 
                         text=text, 
                         provider="anthropic")

# Add route to get available LLM providers (now only Anthropic)
@app.route("/api/llm_providers", methods=["GET"])
def get_llm_providers():
    """Get available LLM providers and their status"""
    providers = {
        "anthropic": {
            "available": ANTHROPIC_AVAILABLE and bool(app.config['ANTHROPIC_API_KEY']),
            "name": "Anthropic Claude",
            "models": [
                "claude-3-haiku-20240307",
                "claude-3-sonnet-20240229", 
                "claude-3-opus-20240229"
            ]
        }
    }
    
    return jsonify(providers)

#################################

# Initialize EasyOCR reader
if not hasattr(app, 'ocr_reader'):
    app.ocr_reader = easyocr.Reader(['en'])

# REDIS connection
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.ping()
except redis.ConnectionError:
    app.logger.error("Cannot connect to Redis. Using in-memory storage.")
    redis_client = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def generate_preview(pdf_path, output_dir, prefix):
    previews = []
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            if i >= app.config['MAX_PREVIEW_PAGES']:
                break
            pix = page.get_pixmap()
            sanitized_prefix = re.sub(r'[^a-zA-Z0-9_-]', '', prefix)
            preview_path = os.path.join(output_dir, f"{sanitized_prefix}_page_{i+1}.png")
            pix.save(preview_path)
            previews.append(preview_path)
        return previews
    except Exception as e:
        app.logger.error(f"Preview generation failed: {str(e)}")
        return []

def to_url_path(system_path):
    return system_path.replace('\\', '/')

#############################################################
import random
import time
from collections import defaultdict, Counter

def get_balanced_samples(data, target_total_samples=100, samples_per_label=None, random_seed=None):
    """
    Get balanced random samples ensuring equal representation from each label.
   
    Args:
        data: List of examples with entities
        target_total_samples: Target total number of samples (default: 500)
        samples_per_label: Fixed number per label (if None, will calculate dynamically)
        random_seed: Random seed (if None, uses timestamp)
    """
    # Use current timestamp if no seed provided (different each time)
    if random_seed is None:
        random_seed = int(time.time() * 1000000) % 2147483647  # Large random number
   
    random.seed(random_seed)
    print(f"Using random seed: {random_seed}")
   
    # Group data by labels (check all entities in each example)
    label_groups = defaultdict(list)
    for item in data:
        # Get all unique labels from entities in this example
        example_labels = set()
        for entity in item.get("entities", []):
            label = entity.get("label")
            if label:
                example_labels.add(label)
       
        # Add this example to all its label groups
        for label in example_labels:
            label_groups[label].append(item)
   
    # Print label distribution
    print("Original label distribution:")
    for label, items in label_groups.items():
        print(f"  {label}: {len(items)} samples")
   
    # Calculate samples per label dynamically
    unique_labels = list(label_groups.keys())
    num_labels = len(unique_labels)
   
    if num_labels == 0:
        print("No labels found in data!")
        return []
   
    # Calculate samples per label if not provided
    if samples_per_label is None:
        samples_per_label = max(1, target_total_samples // num_labels)
        print(f"Calculated {samples_per_label} samples per label for {num_labels} labels (targeting {target_total_samples} total)")
   
    # Check if we can actually get the target number
    total_possible = sum(min(len(items), samples_per_label) for items in label_groups.values())
    if total_possible < target_total_samples:
        print(f"Warning: Can only get {total_possible} samples (less than target {target_total_samples})")
        # Adjust samples per label to use all available data
        max_possible_per_label = max(len(items) for items in label_groups.values())
        samples_per_label = min(samples_per_label, max_possible_per_label)
   
    balanced_samples = []
    used_examples = set()  # Track which examples we've already selected
   
    # Sample from each label
    for label in unique_labels:
        available_samples = [item for item in label_groups[label]
                           if id(item) not in used_examples]
       
        if len(available_samples) >= samples_per_label:
            # Random sample without replacement
            selected = random.sample(available_samples, samples_per_label)
        else:
            # If not enough samples, take all available
            selected = available_samples
            print(f"Warning: Only {len(available_samples)} samples available for label '{label}', requested {samples_per_label}")
       
        # Add to balanced samples and mark as used
        for item in selected:
            if id(item) not in used_examples:
                balanced_samples.append(item)
                used_examples.add(id(item))
       
        print(f"Selected {len([s for s in selected if id(s) not in used_examples])} new samples for label '{label}'")
   
    # Shuffle the final result
    random.shuffle(balanced_samples)
   
    print(f"\nFinal balanced dataset: {len(balanced_samples)} samples")
   
    # Verify balance - count examples per label
    final_counts = defaultdict(int)
    for item in balanced_samples:
        for entity in item.get("entities", []):
            label = entity.get("label")
            if label:
                final_counts[label] += 1
   
    print("Final label distribution (entity counts):")
    for label, count in final_counts.items():
        print(f"  {label}: {count} entities")
   
    return balanced_samples

def measure_gliner_response_time(model, text, labels, threshold=0.7):
    """
    Measure GLiNER model response time for entity prediction
   
    Args:
        model: GLiNER model instance
        text: Input text for entity recognition
        labels: List of labels to detect
        threshold: Confidence threshold
   
    Returns:
        tuple: (entities, response_time_ms, metadata)
    """
    start_time = time.perf_counter()
   
    try:
        entities = model.predict_entities(text, labels, threshold=threshold)
        end_time = time.perf_counter()
       
        response_time_ms = (end_time - start_time) * 1000
       
        metadata = {
            'text_length': len(text),
            'num_labels': len(labels),
            'num_entities_found': len(entities),
            'threshold': threshold,
            'timestamp': datetime.now().isoformat()
        }
       
        return entities, response_time_ms, metadata
       
    except Exception as e:
        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000
        raise Exception(f"GLiNER prediction failed after {response_time_ms:.2f}ms: {str(e)}")

@app.route('/evaluate')
def evaluate():
    try:
        app.logger.info("Starting evaluation...")
       
        # Check if the JSON file exists
        json_file_path = "gliner_pii.json"
        if not os.path.exists(json_file_path):
            app.logger.error(f"Evaluation file {json_file_path} not found")
            return render_template_string("""
            <html>
            <head><title>Evaluation Error</title></head>
            <body>
                <h1>Error</h1>
                <p>Evaluation dataset file 'gliner_pii.json' not found.</p>
                <p>Please ensure the file exists in the application directory.</p>
                <a href="/" style="background: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Back to Home</a>
            </body>
            </html>
            """)
       
        # Load the dataset
        app.logger.info("Loading evaluation dataset...")
        with open(json_file_path, "r", encoding="utf-8") as f:
            gold_data = json.load(f)
       
        app.logger.info(f"Loaded {len(gold_data)} examples from dataset")
       
        # Get balanced samples with different random seed each time
        app.logger.info("Creating balanced sample...")
        balanced_samples = get_balanced_samples(
            data=gold_data,
            target_total_samples=100,  # Target 500 samples total
            random_seed=None  # This will generate a new random seed each time
        )
       
        sample_size = len(balanced_samples)
        app.logger.info(f"Using {sample_size} balanced examples for evaluation")
       
        # Load the model
        app.logger.info("Loading GLiNER model...")
        model = GLiNER.from_pretrained("E3-JSI/gliner-multi-pii-domains-v1")
       
        labels = ["person", "email", "phone", "address", "ssn", "credit_card"]
        label_map = {
            "person": "person",
            "email": "email",
            "phone": "phone",
            "address": "address",
            "ssn": "ssn",
            "credit_card": "credit_card",
        }

        def fuzzy_match_span(gold_span, pred_spans, tolerance=2):
            g_start, g_end, g_label = gold_span
            for p_start, p_end, p_label in pred_spans:
                if g_label == p_label and abs(g_start - p_start) <= tolerance and abs(g_end - p_end) <= tolerance:
                    return (p_start, p_end, p_label)
            return None

        metrics = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
        total_tp = total_fp = total_fn = 0
        debug_count = 0

        app.logger.info("Starting evaluation loop...")

        for i, example in enumerate(balanced_samples):
            if i % 10 == 0:
                app.logger.info(f"Processed {i}/{sample_size} examples")
               
            text = example["text"]
            gold_spans = {(ent["start"], ent["end"], ent["label"]) for ent in example["entities"]}
           
            try:
                pred_entities, eval_response_time, eval_metadata = measure_gliner_response_time(model, text, labels, threshold=0.6)
                if i % 10 == 0:  # Log every 10th evaluation
                    print(f"GLiNER Evaluation time {i}: {eval_response_time:.2f}ms for {eval_metadata['text_length']} chars")
                pred_spans = {
                    (ent["start"], ent["end"], label_map.get(ent["label"].lower(), ent["label"].lower()))
                    for ent in pred_entities
                }
            except Exception as e:
                app.logger.error(f"Error predicting entities for example {i}: {str(e)}")
                continue

            matched_preds = set()
            for g in gold_spans:
                matched = fuzzy_match_span(g, pred_spans)
                if matched:
                    metrics[g[2]]["tp"] += 1
                    matched_preds.add(matched)
                    total_tp += 1
                else:
                    metrics[g[2]]["fn"] += 1
                    total_fn += 1

                    if g[2] in {"person", "misc"} and debug_count < 5:
                        app.logger.info(f"Missed entity: {g} in text: {text[:100]}...")
                        debug_count += 1

            for p in pred_spans:
                if p not in matched_preds:
                    if p[2] in metrics:  # Only count if it's a valid label
                        metrics[p[2]]["fp"] += 1
                    total_fp += 1

        app.logger.info("Calculating final metrics...")
       
        final_metrics = {}
        for label in labels:
            tp, fp, fn = metrics[label]["tp"], metrics[label]["fp"], metrics[label]["fn"]
            p = tp / (tp + fp + 1e-10)
            r = tp / (tp + fn + 1e-10)
            f = 2 * p * r / (p + r + 1e-10)
            accuracy = tp / (tp + fp + fn + 1e-10)
            final_metrics[label] = {"precision": p, "recall": r, "f1": f, "accuracy": accuracy}

        micro_p = total_tp / (total_tp + total_fp + 1e-10)
        micro_r = total_tp / (total_tp + total_fn + 1e-10)
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-10)
        micro_accuracy = total_tp / (total_tp + total_fp + total_fn + 1e-10)

        final_metrics["micro"] = {
            "precision": micro_p,
            "recall": micro_r,
            "f1": micro_f1,
            "accuracy": micro_accuracy
        }
       
        app.logger.info("Evaluation completed successfully")
       
        return render_template("evaluate.html", metrics=final_metrics, num_examples=sample_size)
   
    except Exception as e:
        app.logger.error(f"Evaluation error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return render_template_string(f"""
        <html>
        <head><title>Evaluation Error</title></head>
        <body>
            <h1>Evaluation Error</h1>
            <p>An error occurred during evaluation:</p>
            <pre>{str(e)}</pre>
            <a href="/" style="background: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Back to Home</a>
        </body>
        </html>
        """)

#######################################################
def apply_blur_effect(page, rect):
    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    ).copy()
    r = rect * zoom
    x0, y0, x1, y1 = map(int, [r.x0, r.y0, r.x1, r.y1])
    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
        return
    roi = img[y0:y1, x0:x1]
    blurred_roi = cv2.GaussianBlur(roi, (25, 25), 0)
    img[y0:y1, x0:x1] = blurred_roi
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    blurred_pix = fitz.Pixmap(buffer)
    page.clean_contents()
    page.insert_image(page.rect, pixmap=blurred_pix)

def apply_blackout(page, rect):
    annot = page.add_redact_annot(rect, fill=(0, 0, 0))
    annot.update()

def redact_image(pil_image, boxes, style='blur'):
    """Redact detected PII in image by blurring or blacking out regions"""
    img_np = np.array(pil_image)
    for box in boxes:
        x_min, y_min = map(int, box[0])
        x_max, y_max = map(int, box[2])
        roi = img_np[y_min:y_max, x_min:x_max]
       
        if style == 'blur':
            # Apply Gaussian blur
            blurred_roi = cv2.GaussianBlur(roi, (25, 25), 0)
            img_np[y_min:y_max, x_min:x_max] = blurred_roi
        else:  # blackout
            # Fill with black
            img_np[y_min:y_max, x_min:x_max] = [0, 0, 0]
           
    return Image.fromarray(img_np)

def extract_image_text(image_bytes):
    """Extract text and bounding boxes from image"""
    try:
        pil_image = Image.open(BytesIO(image_bytes))
        img_np = np.array(pil_image)
        results = app.ocr_reader.readtext(img_np, detail=1)
        return results
    except Exception as e:
        logging.error(f"OCR failed: {str(e)}")
        return []

def is_valid_pii(entity, entity_text, pii_label):
    """Validate detected PII with type-specific pattern checks"""
    # Normalize label to base form
    base_label = pii_label.lower().split('_')[-1]
   
    # Define regex patterns for structured PII types
    pii_patterns = {
        'ssn': r'^\d{3}-?\d{2}-?\d{4}$',
        'creditcard': r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$',
        'phone': r'^(\+?\d{1,3}[-\s.]?)?(\(?\d{1,4}\)?[-\s.]?)?(\d{1,4}[-\s.]?){1,3}(?:ext\.?\s?\d+|x\d+)?$',
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'dateofbirth': r'^(0?[1-9]|1[0-2])/(0?[1-9]|[12][0-9]|3[01])/\d{4}$',
        'driverslicense': r'^[A-Z0-9]{5,15}$',
        'passport': r'^[A-Z0-9]{8,12}$',
        'bankaccount': r'^\d{10,17}$',
        'ip_address': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
        'ipv6': r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$',
        'cvv': r'^\d{3,4}$',
        'credit_card_expiration': r'^(0[1-9]|1[0-2])\/?([0-9]{2})$',
        'iban': r'^[A-Z]{2}\d{2}[A-Z0-9]{4,30}$',
        'cpf': r'^\d{3}\.?\d{3}\.?\d{3}-?\d{2}$',  # Brazilian CPF
        'cnpj': r'^\d{2}\.?\d{3}\.?\d{3}\/?\d{4}-?\d{2}$',  # Brazilian CNPJ
        'national_id': r'^[A-Z0-9]{5,20}$',
        'health_insurance_number': r'^[A-Z]{3}\d{6}$',
        'medical_record_number': r'^[A-Z0-9]{8,12}$',
        'biometric_identifier': r'^[A-Z0-9]{16}$',
        'license_plate': r'^[A-Z0-9]{2,10}$',
        'coordinates': r'^-?\d{1,3}\.\d{6},\s*-?\d{1,3}\.\d{6}$',
        'swift_bic': r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$',
        'api_key': r'^[a-z0-9]{32}$',
        'digital_signature': r'^[A-Z0-9+/]{40,}$',
        'visa_number': r'^[A-Z0-9]{8,12}$',
        'blood_type': r'^(A|B|AB|O)[+-]$'
    }
   
    # Skip validation for non-pattern based types
    if base_label not in pii_patterns:
        return True
   
    # Apply regex validation
    pattern = pii_patterns[base_label]
    return re.match(pattern, entity_text) is not None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    files = request.files.getlist('file')
    if not files or any(file.filename == '' for file in files):
        return 'No selected file', 400

    files = [file for file in files if allowed_file(file.filename)]
    if not files:
        return 'Invalid file type', 400

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    selected_pii = request.form.getlist('pii_types')
    pii_labels = [label.strip().lower().replace(' ', '') for label in selected_pii if label.strip()]
    if not pii_labels:
        return "No PII types selected", 400
    redaction_style = request.form.get('redaction_style', 'blur')

    redis_key = f"redaction_results:{timestamp}"

    for file in files:
        original_name = secure_filename(file.filename.rsplit('.', 1)[0])
        input_filename = f"{timestamp}_{original_name}.pdf"
        output_filename = f"redacted_{timestamp}_{original_name}.pdf"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

        try:
            file.save(input_path)
            model = GLiNER.from_pretrained("E3-JSI/gliner-multi-pii-domains-v1")

            with fitz.open(input_path) as doc:
                if not doc.is_pdf:
                    raise ValueError("Invalid PDF file")

                for page in doc:
                    full_text = page.get_text() or ""
                    image_redaction_needed = False
                   
                    # Process images in PDF
                    image_list = page.get_images()
                    for img_index, img in enumerate(image_list):
                        base_image = doc.extract_image(img[0])
                        image_bytes = base_image["image"]
                       
                        # Extract text and bounding boxes from image
                        ocr_results = extract_image_text(image_bytes)
                        if not ocr_results:
                            continue
                           
                        # Combine OCR text for PII detection
                        image_text = " ".join([res[1] for res in ocr_results])
                        full_text += f"\n[IMAGE_TEXT]: {image_text}"
                       
                        # Detect PII in image text
                        image_entities, img_response_time, img_metadata = measure_gliner_response_time(model, image_text, pii_labels, threshold=0.7)
                        print(f"GLiNER image time: {img_response_time:.2f}ms for {img_metadata['text_length']} chars, found {img_metadata['num_entities_found']} entities")
                        valid_image_entities = []
                        for ent in image_entities:
                            if is_valid_pii(ent, ent['text'], ent['label']):
                                valid_image_entities.append(ent)
                       
                        if valid_image_entities:
                            image_redaction_needed = True
                            pii_boxes = []
                            for ent in valid_image_entities:
                                # Find OCR results matching the PII text
                                for res in ocr_results:
                                    if ent['text'] in res[1]:
                                        pii_boxes.append(res[0])
                           
                            # Redact image if PII found
                            if pii_boxes:
                                pil_image = Image.open(BytesIO(image_bytes))
                                redacted_image = redact_image(pil_image, pii_boxes, style=redaction_style)
                               
                                # Convert to bytes
                                img_byte_arr = BytesIO()
                                redacted_image.save(img_byte_arr, format='PNG')
                                redacted_bytes = img_byte_arr.getvalue()
                               
                                # Replace image in PDF
                                page.replace_image(img[0], pixmap=fitz.Pixmap(redacted_bytes))

                    # Process digital text
                    if full_text.strip():
                        try:
                            entities, text_response_time, text_metadata = measure_gliner_response_time(model, full_text, pii_labels, threshold=0.7)
                            print(f"GLiNER text time: {text_response_time:.2f}ms for {text_metadata['text_length']} chars, found {text_metadata['num_entities_found']} entities")
                        except Exception as e:
                            app.logger.error(f"GLiNER model prediction failed: {str(e)}")
                            entities = []
                    else:
                        entities = []

                    # Filter entities with validation
                    valid_entities = []
                    for ent in entities:
                        entity_text = full_text[ent['start']:ent['end']].strip()
                        if is_valid_pii(ent, entity_text, ent['label']):
                            valid_entities.append(ent)
                   
                    entities = valid_entities

                    # Process and redact digital text
                    merged_entities = []
                    if entities:
                        entities = sorted(entities, key=lambda x: x['start'])
                        filtered_entities = []
                        for ent in entities:
                            entity_text = full_text[ent['start']:ent['end']].strip()
                            if (entity_text == 'I' or
                               (len(entity_text) < 2 and not entity_text.isupper()) or
                               ent['score'] < 0.5):
                                continue
                            filtered_entities.append(ent)
                       
                        if not filtered_entities:
                            continue
                       
                        current = filtered_entities[0]
                        for next_ent in filtered_entities[1:]:
                            if (current['label'] == next_ent['label'] and
                                next_ent['start'] <= current['end'] + 1):
                                current['end'] = next_ent['end']
                                current['text'] = full_text[current['start']:current['end']].strip()
                                current['score'] = max(current['score'], next_ent['score'])
                            else:
                                merged_entities.append(current)
                                current = next_ent
                        merged_entities.append(current)

                    # Apply digital text redaction
                    for entity in merged_entities:
                        entity_text = entity['text'].strip()
                        rects = page.search_for(entity_text)
                        for rect in rects:
                            if redaction_style == 'blur':
                                apply_blur_effect(page, rect)
                            else:  # blackout
                                apply_blackout(page, rect)
                   
                    # Apply all blackout annotations at once
                    if redaction_style == 'blackout':
                        page.apply_redactions()

                doc.save(output_path, garbage=3, deflate=True)

            result_data = {
                'original': input_filename,
                'redacted': output_filename,
                'status': 'processed'
            }
            if redis_client:
                redis_client.rpush(redis_key, json.dumps(result_data))
            else:
                if not hasattr(app, 'in_memory_results'):
                    app.in_memory_results = {}
                if redis_key not in app.in_memory_results:
                    app.in_memory_results[redis_key] = []
                app.in_memory_results[redis_key].append(result_data)

        except Exception as e:
            app.logger.error(f"Error processing {file.filename}: {str(e)}")
            result_data = {
                'original': file.filename,
                'error': str(e),
                'status': 'failed'
            }
            if redis_client:
                redis_client.rpush(redis_key, json.dumps(result_data))
            else:
                if not hasattr(app, 'in_memory_results'):
                    app.in_memory_results = {}
                if redis_key not in app.in_memory_results:
                    app.in_memory_results[redis_key] = []
                app.in_memory_results[redis_key].append(result_data)

    return redirect(url_for('get_results', timestamp=timestamp))

@app.route('/results/<timestamp>')
def get_results(timestamp):
    redis_key = f"redaction_results:{timestamp}"
    results = []
    if redis_client:
        for i in range(redis_client.llen(redis_key)):
            result_data = json.loads(redis_client.lindex(redis_key, i))
            results.append(result_data)
    else:
        results = app.in_memory_results.get(redis_key, [])
    return render_template('preview_multiple.html',
                          results=results,
                          timestamp=timestamp)

@app.route('/results/<timestamp>/<file_id>')
def get_file_result(timestamp, file_id):
    redis_key = f"redaction_results:{timestamp}"
    results = []
    if redis_client:
        for i in range(redis_client.llen(redis_key)):
            result_data = json.loads(redis_client.lindex(redis_key, i))
            results.append(result_data)
    else:
        results = app.in_memory_results.get(redis_key, [])

    result = next((r for r in results if r['original'] == file_id), None)
    if not result:
        return jsonify({"status": "processing"}), 404

    if result.get('status') == 'processed':
        preview_dir = os.path.join(app.config['PREVIEW_FOLDER'], timestamp)
        os.makedirs(preview_dir, exist_ok=True)
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], result['original'])
        redacted_path = os.path.join(app.config['PROCESSED_FOLDER'], result['redacted'])
        original_base = result['original'].rsplit('.', 1)[0]
        redacted_base = result['redacted'].rsplit('.', 1)[0]
        original_previews = generate_preview(original_path, preview_dir, f"original_{original_base}")
        redacted_previews = generate_preview(redacted_path, preview_dir, f"redacted_{redacted_base}")
        preview = {
            'original': [to_url_path(os.path.relpath(p, 'static')) for p in original_previews],
            'redacted': [to_url_path(os.path.relpath(p, 'static')) for p in redacted_previews],
            'download_url': url_for('download_file', filename=result['redacted'])
        }
    else:
        preview = {'original': [], 'redacted': [], 'download_url': '#'}
    return jsonify({
        'status': result['status'],
        'previews': preview,
        'original': result['original'],
        'error': result.get('error', '')
    })

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    return send_file(
        path,
        as_attachment=True,
        download_name=f"redacted_{filename.split('_', 2)[-1]}",
        mimetype='application/pdf'
    )

def cleanup_old_files():
    now = time.time()
    for root, dirs, files in os.walk(app.config['PREVIEW_FOLDER']):
        for file in files:
            file_path = os.path.join(root, file)
            if os.stat(file_path).st_mtime < now - 300:
                try:
                    os.remove(file_path)
                except Exception as e:
                    app.logger.error(f"Failed to delete preview file {file_path}: {e}")
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                try:
                    os.rmdir(dir_path)
                except Exception as e:
                    app.logger.error(f"Failed to delete empty directory {dir_path}: {e}")
    for root, dirs, files in os.walk(app.config['PROCESSED_FOLDER']):
        for file in files:
            file_path = os.path.join(root, file)
            if os.stat(file_path).st_mtime < now - 300:
                try:
                    os.remove(file_path)
                except Exception as e:
                    app.logger.error(f"Failed to delete processed file {file_path}: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_old_files, 'interval', minutes=5)
scheduler.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)