# Carbonate Microfacies Interpreter
#Author: Dr. CarbonateGPT (2024)
# An AI-powered tool for interpreting carbonate microfacies (SMFs) and their associated depositional environments (FZs) 
# based on textual descriptions of thin sections. Utilizes a bi-encoder for retrieval and a cross-encoder for fine-grained ranking,
# grounded in the Flügel (2010) classification system.

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, CrossEncoder, util

class GranularFaciesInterpreter:
    def __init__(self):
        print("Initializing Digital Carbonate Geologist...")

        self.SMF_TYPES = {
            'SMF1': 'Spiculite wackestone/packstone',
            'SMF2': 'Microbioclastic peloidal calcisiltite',
            'SMF3': 'Pelagic mudstone/wackestone',
            'SMF4': 'Microbreccia, bio-lithoclastic packstone',
            'SMF5': 'Allochthonous bioclastic grainstone/rudstone',
            'SMF6': 'Densely packed reef rudstone',
            'SMF7': 'Organic boundstone, platform-margin reef',
            'SMF8': 'Whole fossil wackestone/floatstone',
            'SMF9': 'Burrowed bioclastic wackestone',
            'SMF10': 'Bioclastic packstone/wackestone with worn grains',
            'SMF11': 'Coated bioclastic grainstone',
            'SMF12S': 'Limestone with shell concentrations',
            'SMF12C': 'Limestone with crinoid concentrations',
            'SMF13': 'Oncoid rudstone/grainstone',
            'SMF14': 'Lag deposits',
            'SMF15C': 'Ooid grainstone with concentric ooids',
            'SMF15M': 'Ooid grainstone with micritic ooids',
            'SMF15R': 'Ooid grainstone with radial ooids',
            'SMF16P': 'Non-laminated peloidal grainstone/packstone',
            'SMF16L': 'Laminated peloidal bindstone',
            'SMF17': 'Aggregate-grain grainstone',
            'SMF18': 'Grainstone/packstone with foraminifera/algae',
            'SMF19': 'Densely laminated bindstone',
            'SMF20': 'Laminated stromatolitic boundstone/mudstone',
            'SMF21': 'Fenestral packstone/bindstone',
            'SMF22': 'Oncoid floatstone/packstone',
            'SMF23': 'Homogeneous, non-fossiliferous micrite',
            'SMF24': 'Lithoclastic floatstone/rudstone/breccia',
            'SMF25': 'Laminated evaporite-carbonate mudstone',
            'SMF26': 'Pisoid cementstone/rudstone/packstone'
        }

        self.FZ_TYPES = {
            'FZ1': 'Deep water basin', 'FZ2': 'Deep-shelf', 'FZ3': 'Toe-of-slope',
            'FZ4': 'Slope', 'FZ5': 'Platform margin-reef', 'FZ6': 'Platform margin-sand shoal',
            'FZ7': 'Open marine platform', 'FZ8': 'Restricted platform',
            'FZ9': 'Evaporitic/brackish', 'FZ10': 'Meteorically restricted'
        }

        self.SMF_FZ_MAP = {
            'SMF1': ['FZ1', 'FZ2'], 'SMF2': ['FZ1', 'FZ2', 'FZ3'], 'SMF3': ['FZ1', 'FZ3'],
            'SMF4': ['FZ1', 'FZ3', 'FZ4'], 'SMF5': ['FZ4'], 'SMF6': ['FZ4'], 'SMF7': ['FZ5'],
            'SMF8': ['FZ2', 'FZ7'], 'SMF9': ['FZ2', 'FZ7'], 'SMF10': ['FZ2', 'FZ7'],
            'SMF11': ['FZ5', 'FZ6'], 'SMF12S': ['FZ1', 'FZ2', 'FZ3', 'FZ4', 'FZ5', 'FZ6', 'FZ7', 'FZ8'],
            'SMF12C': ['FZ2', 'FZ4', 'FZ5'], 'SMF13': ['FZ5', 'FZ6', 'FZ7'],
            'SMF14': ['FZ3', 'FZ4', 'FZ5', 'FZ7'], 'SMF15C': ['FZ6', 'FZ7'],
            'SMF15M': ['FZ8', 'FZ9'], 'SMF15R': ['FZ8', 'FZ9'], 'SMF16P': ['FZ8'],
            'SMF16L': ['FZ4', 'FZ5', 'FZ7', 'FZ9'], 'SMF17': ['FZ7', 'FZ8'],
            'SMF18': ['FZ7', 'FZ8'], 'SMF19': ['FZ8', 'FZ9'],
            'SMF20': ['FZ7', 'FZ8', 'FZ9'], 'SMF21': ['FZ8', 'FZ9'],
            'SMF22': ['FZ8'], 'SMF23': ['FZ8', 'FZ9'], 'SMF24': ['FZ8'],
            'SMF25': ['FZ9'], 'SMF26': ['FZ10']
        }

        # Load Models
        self.bi_encoder = SentenceTransformer('all-mpnet-base-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Embed Knowledge Base
        self.smf_keys = list(self.SMF_TYPES.keys())
        self.search_texts = [f"{k}: {v}" for k, v in self.SMF_TYPES.items()]
        self.corpus_embeddings = self.bi_encoder.encode(self.search_texts, convert_to_tensor=True)
        print("System Ready.")

    def predict_facies(self, description):
        # Stage 1: Retrieval
        query_embedding = self.bi_encoder.encode(description, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=5)[0]

        # Stage 2: Re-ranking
        candidate_pairs = []
        for hit in hits:
            idx = hit['corpus_id']
            candidate_pairs.append([description, self.search_texts[idx]])

        logits = self.cross_encoder.predict(candidate_pairs)

        # Stage 3: Softmax
        logits_tensor = torch.tensor(logits)
        probs = F.softmax(logits_tensor, dim=0).tolist()

        ranked_results = []
        for i, prob in enumerate(probs):
            hit_idx = hits[i]['corpus_id']
            smf_key = self.smf_keys[hit_idx]
            ranked_results.append({
                "SMF": smf_key,
                "Description": self.SMF_TYPES[smf_key],
                "Confidence": prob
            })

        ranked_results = sorted(ranked_results, key=lambda x: x['Confidence'], reverse=True)
        best_match = ranked_results[0]

        # Stage 4: Mapping
        associated_fzs = self.SMF_FZ_MAP.get(best_match['SMF'], [])
        fz_details = [f"**{fz}**: {self.FZ_TYPES.get(fz, 'Unknown')}" for fz in associated_fzs]

        return {
            "Predicted_SMF": best_match['SMF'],
            "SMF_Description": best_match['Description'],
            "Confidence": best_match['Confidence'],
            "Facies_Zones": fz_details,
            "Alternatives": ranked_results # Pass all for the chart
        }


interpreter = GranularFaciesInterpreter()

# gradio
def interpret_microfacies(text):
    if not text or len(text.strip()) < 5:
        return None, "Please enter a valid description."

    result = interpreter.predict_facies(text)

    # Output 1: Confidence Dictionary for the Label (Bar Chart)
    # We create a dict of {Label: Confidence} for the top 5
    conf_dict = {item['SMF']: item['Confidence'] for item in result['Alternatives']}

    # Output 2: Geological Report (Markdown)
    report = f"""
    ### Primary Interpretation: {result['Predicted_SMF']}
    _{result['SMF_Description']}_

    ---
    #### 🌍 Depositional Environment (Facies Zones)
    The model suggests the following likely environments based on Flügel (2010):

    {chr(10).join(['- ' + fz for fz in result['Facies_Zones']])}

    ---
    #### 🧠 AI Reasoning
    The Cross-Encoder analyzed the specific features in your text (e.g., grain types, matrix, lamination) to distinguish this specific SMF from texturally similar alternatives.
    """

    return conf_dict, report

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔬 Carbonate Microfacies Interpreter
        **AI-Powered Geological Reasoning for Standard Microfacies/Facies Zone Identification**: Paste a description of a thin section below.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                lines=6,
                placeholder="e.g., Grainstone composed of well-sorted ooids with concentric cortical laminations. No mud matrix...",
                label="Thin Section Description"
            )
            submit_btn = gr.Button("Interpret Microfacies", variant="primary")

            # Example Inputs
            gr.Examples(
                examples=[
                    "Grainstone composed of well-sorted ooids with concentric cortical laminations. No mud matrix.",
                    "Packstone texture dominated by peloids. Irregular, mm-scale voids (fenestrae) are common. No lamination.",
                    "Wackestone with abundant whole fossils including gastropods and brachiopods. Matrix is micrite.",
                    "Grainstone with aggregate grains and grapestones cemented together."
                ],
                inputs=input_text
            )

        with gr.Column(scale=1):
            # Output 1: Bar Chart
            label_output = gr.Label(num_top_classes=5, label="SMF Probability Distribution")

            # Output 2: Detailed Report
            report_output = gr.Markdown(label="Geological Report")

    # Connect logic
    submit_btn.click(
        fn=interpret_microfacies,
        inputs=input_text,
        outputs=[label_output, report_output]
    )

if __name__ == "__main__":
    demo.launch()