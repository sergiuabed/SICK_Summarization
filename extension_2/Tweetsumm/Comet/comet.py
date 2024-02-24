import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from Comet.utils import use_task_specific_params, trim_batch


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(
        self,
        queries,
        decode_method="beam",
        num_generate=5,
    ):

        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(
                    batch, return_tensors="pt", truncation=True, padding="max_length"
                ).to(self.device)
                input_ids, attention_mask = trim_batch(
                    **batch, pad_token_id=self.tokenizer.pad_token_id
                )

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                )

                dec = self.tokenizer.batch_decode(
                    summaries,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                decs.append(dec)

            return decs


def generate_commonsense(comet, utterance, num_generate=5):
    rels = ["HinderedBy", "xWant", "xIntent", "xNeed", "xReason"]
    data = {}
    data["sentence"] = utterance
    data["speaker"] = utterance.split()[0]
    for rel in rels:
        query = f"{utterance} {rel} [GEN]"
        results = comet.generate(
            [query], decode_method="beam", num_generate=num_generate
        )
        data[rel] = results[0]
    return data


all_relations = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
]
