from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import json


docs_json = json.load(open("cranfield/cran_docs.json", 'r'))[:]
doc_ids, raw_docs = [item["id"] for item in docs_json], \
                        [item["body"] for item in docs_json]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(raw_docs)

svd = TruncatedSVD(n_components=1300, random_state=42)

newX = svd.fit_transform(X)

print(svd.explained_variance_ratio_.sum())