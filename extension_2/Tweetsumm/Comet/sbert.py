from sentence_transformers import SentenceTransformer, util


def select_best_commonsense(model, data, debug=False):
    utterance = data["sentence"]
    embedded_utterance = model.encode(utterance, convert_to_tensor=True)
    commonsenses = []
    commonsenses.extend(data["HinderedBy"])
    commonsenses.extend(data["xWant"])
    commonsenses.extend(data["xIntent"])
    commonsenses.extend(data["xNeed"])
    commonsenses.extend(data["xReason"])
    embedded_commonsenses = model.encode(commonsenses, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embedded_utterance, embedded_commonsenses)

    if debug:
        for i in range(len(embedded_commonsenses)):
            print(
                "{} \t\t {} \t Score: {:.4f}".format(
                    utterance, commonsenses[i], cosine_scores[0][i]
                )
            )

    best_commonsense = commonsenses[cosine_scores.argmax()]
    best_relation = ""
    for relation in data:
        if best_commonsense in data[relation]:
            best_relation = relation
            break
    return {
        "sentence":utterance,
        "relation":best_relation,
        "commonsense":best_commonsense
    }
