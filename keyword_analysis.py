import re
from sklearn.feature_extraction.text import CountVectorizer

def extract_keywords(text, top_n=10):
    """
    Εξάγει τις πιο συχνές λέξεις-κλειδιά από ένα κείμενο, αφαιρώντας stop words και ειδικούς χαρακτήρες.

    Args:
        text (str): Το κείμενο προς ανάλυση.
        top_n (int): Ο αριθμός των κορυφαίων λέξεων-κλειδιών που θα εξαγάγουμε.

    Returns:
        List[str]: Οι πιο συχνές λέξεις-κλειδιά.
    """
    # Καθαρισμός κειμένου
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())  # Αφαιρούμε ειδικούς χαρακτήρες
    words = text.split()

    # Χρήση CountVectorizer για τον εντοπισμό των πιο συχνών λέξεων
    vectorizer = CountVectorizer(stop_words="english", max_features=top_n)
    word_counts = vectorizer.fit_transform([" ".join(words)])
    keywords = vectorizer.get_feature_names_out()

    return list(keywords)

def find_common_keywords(datasets, top_n=10):
    """
    Εντοπίζει κοινές λέξεις-κλειδιά μεταξύ πολλαπλών datasets.

    Args:
        datasets (List[str]): Λίστα με τα datasets ως κείμενα.
        top_n (int): Ο αριθμός των κορυφαίων λέξεων-κλειδιών που θα εξεταστούν.

    Returns:
        Dict: Ένα λεξικό που δείχνει τις κοινές λέξεις-κλειδιά μεταξύ των datasets.
    """
    keyword_sets = []

    for dataset in datasets:
        keywords = extract_keywords(dataset, top_n)
        keyword_sets.append(set(keywords))

    common_keywords = set.intersection(*keyword_sets) if len(keyword_sets) > 1 else set()

    # Δημιουργία αποτελέσματος
    results = {
        "common_keywords": list(common_keywords),
        "individual_keywords": {f"Dataset {i+1}": list(keywords) for i, keywords in enumerate(keyword_sets)}
    }

    return results