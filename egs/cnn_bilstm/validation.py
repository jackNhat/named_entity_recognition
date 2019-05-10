def compute_f1(predictions, correct, idx2_label):
    label_pred = []
    for sentence in predictions:
        label_pred.append([idx2_label[element] for element in sentence])

    label_correct = []
    for sentence in correct:
        label_correct.append([idx2_label[element] for element in sentence])

    prec = compute_precision(label_pred, label_correct)
    rec = compute_precision(label_correct, label_pred)

    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)

    return prec, rec, f1


def compute_precision(guessed_sentences, correct_sentences):
    assert (len(guessed_sentences) == len(correct_sentences))
    correct_count = 0
    count = 0

    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        assert (len(guessed) == len(correct))
        idx = 0
        while idx < len(guessed):
            if guessed[idx][0] == 'B':
                count += 1

                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctly_found = True

                    while idx < len(guessed) and guessed[idx][0] == 'I':
                        if guessed[idx] != correct[idx]:
                            correctly_found = False

                        idx += 1

                    if idx < len(guessed):
                        if correct[idx][0] == 'I':
                            correctly_found = False

                    if correctly_found:
                        correct_count += 1
                else:
                    idx += 1
            else:
                idx += 1

    precision = 0
    if count > 0:
        precision = float(correct_count) / count

    return precision
