import os
import re
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import warnings
warnings.filterwarnings("ignore")

EMBED_SIZE = 300
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

train_files = {
    "en_positive": r"C:\Users\30731\Desktop\Englishsample.positive.txt",
    "en_negative": r"C:\Users\30731\Desktop\Englishsample.negative.txt",
    "zh_positive": r"C:\Users\30731\Desktop\Chinesesample.positive.txt",
    "zh_negative": r"C:\Users\30731\Desktop\Chinesesample.negative.txt"
}

test_files_labeled = {
    "en_test_labeled": r"C:\Users\30731\Desktop\Englishtest.label.txt",
    "zh_test_labeled": r"C:\Users\30731\Desktop\Chinesestest.label.txt"
}

stop_words_file = r"C:\Users\30731\Desktop\cn_stopwords.txt"

def load_stop_words(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return set([w.strip() for w in f if w.strip()])
    return set()

def read_file(path):
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""

def read_reviews(path):
    text = read_file(path)
    return [r.strip() for r in re.findall(r"<review id=\"\d+\">(.*?)</review>", text, re.S) if r.strip()]

def read_labeled_reviews(path):
    text = read_file(path)
    matches = re.findall(r'<review id="\d+"\s+label="(\d+)">(.*?)</review>', text, re.S)
    reviews = [m[1].strip() for m in matches]
    labels = [int(m[0]) for m in matches]
    return reviews, labels

def tokenize(text, lang, stop_words=None):
    if lang == "zh":
        tokens = [w for w in jieba.cut(text) if w.strip()]
        if stop_words:
            tokens = [w for w in tokens if w not in stop_words]
        tokens = [w for w in tokens if re.search(r'[\u4e00-\u9fa5A-Za-z0-9]', w)]
        return tokens
    toks = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
    if stop_words:
        toks = [t for t in toks if t not in stop_words]
    return toks

def build_tfidf(corpus_token_lists):
    docs = [" ".join(tokens) for tokens in corpus_token_lists]
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tfidf.fit(docs)
    feature_names = tfidf.get_feature_names_out()
    idf_vals = tfidf.idf_
    idf_dict = dict(zip(feature_names, idf_vals))
    return tfidf, idf_dict

def sentence_vector(tokens, w2v, idf_dict=None, embed_size=EMBED_SIZE, global_mean=None):
    vecs = []
    weights = []
    for w in tokens:
        if w in w2v.wv:
            vec = w2v.wv[w]
            weight = idf_dict.get(w, 1.0) if idf_dict is not None else 1.0
            vecs.append(vec * weight)
            weights.append(weight)
    if vecs:
        vecs = np.vstack(vecs)
        w_sum = np.sum(weights) if weights else len(vecs)
        return np.sum(vecs, axis=0) / (w_sum + 1e-12)
    if global_mean is not None:
        return global_mean
    return np.zeros(embed_size, dtype=float)

def sentences_to_matrix(tokenized_sentences, w2v, idf_dict=None, embed_size=EMBED_SIZE, global_mean=None):
    X = np.zeros((len(tokenized_sentences), embed_size), dtype=float)
    for i, tokens in enumerate(tokenized_sentences):
        X[i] = sentence_vector(tokens, w2v, idf_dict, embed_size, global_mean)
    return X

if __name__ == "__main__":
    stop_words = load_stop_words(stop_words_file)
    train_tokens_en = []
    train_tokens_zh = []
    train_labels = []

    for key, path in train_files.items():
        if not os.path.exists(path):
            print(f"[WARN] 训练文件不存在: {path}")
            continue
        lang = "zh" if key.startswith("zh") else "en"
        label = 1 if "positive" in key else 0
        raw_reviews = read_reviews(path)
        tokenized = [tokenize(r, lang, stop_words) for r in raw_reviews]
        if lang == "zh":
            train_tokens_zh.extend(tokenized)
        else:
            train_tokens_en.extend(tokenized)
        train_labels.extend([label] * len(tokenized))

    print(f"训练样本总数: {len(train_labels)} (en={len(train_tokens_en)}, zh={len(train_tokens_zh)})")

    models = {}
    if train_tokens_en:
        print("训练 English Word2Vec ...")
        w2v_en = Word2Vec(
            sentences=train_tokens_en,
            vector_size=EMBED_SIZE,
            window=5,
            min_count=1,
            sg=1,
            workers=4,
            epochs=15,
            seed=RANDOM_SEED
        )
        models['en'] = w2v_en
    if train_tokens_zh:
        print("训练 Chinese Word2Vec ...")
        w2v_zh = Word2Vec(
            sentences=train_tokens_zh,
            vector_size=EMBED_SIZE,
            window=5,
            min_count=1,
            sg=1,
            workers=4,
            epochs=15,
            seed=RANDOM_SEED
        )
        models['zh'] = w2v_zh

    idf_en = None
    idf_zh = None
    if train_tokens_en:
        tfidf_en, idf_en = build_tfidf(train_tokens_en)
    if train_tokens_zh:
        tfidf_zh, idf_zh = build_tfidf(train_tokens_zh)

    X_train_list = []
    y_train = np.array(train_labels, dtype=int)
    ordered_tokens = []
    for key, path in train_files.items():
        if not os.path.exists(path):
            continue
        lang = "zh" if key.startswith("zh") else "en"
        raw_reviews = read_reviews(path)
        tokenized = [tokenize(r, lang, stop_words) for r in raw_reviews]
        ordered_tokens.extend([(lang, t) for t in tokenized])

    global_mean = {}
    for lang, w2v in models.items():
        all_vecs = w2v.wv.vectors
        global_mean[lang] = np.mean(all_vecs, axis=0)

    X_train = np.zeros((len(ordered_tokens), EMBED_SIZE), dtype=float)
    for i, (lang, toks) in enumerate(ordered_tokens):
        w2v = models.get(lang)
        idf = idf_zh if lang == 'zh' else idf_en
        gm = global_mean.get(lang)
        X_train[i] = sentence_vector(toks, w2v, idf, EMBED_SIZE, global_mean=gm)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    svm_model = SVC(kernel="linear", C=1.0, class_weight="balanced", probability=False, random_state=RANDOM_SEED)
    print("训练 SVM ...")
    svm_model.fit(X_train, y_train)
    print("SVM 训练完成")

    for key, path in test_files_labeled.items():
        if not os.path.exists(path):
            print(f"[WARN] 测试文件不存在: {path}")
            continue
        lang = "zh" if key.startswith("zh") else "en"
        reviews, labels = read_labeled_reviews(path)
        if not reviews:
            print(f"[WARN] 没有在 {path} 中找到标注样本")
            continue
        tokenized = [tokenize(r, lang, stop_words) for r in reviews]
        w2v = models.get(lang)
        idf = idf_zh if lang == 'zh' else idf_en
        gm = global_mean.get(lang)
        X_test = sentences_to_matrix(tokenized, w2v, idf, EMBED_SIZE, global_mean=gm)
        X_test = scaler.transform(X_test)
        y_test = np.array(labels, dtype=int)
        pred = svm_model.predict(X_test)

        print("\n==============================")
        print(f"SVM - {key}")
        print("总体 Accuracy:", accuracy_score(y_test, pred))
        print("\n分类报告 (labels: 1=正面, 0=负面):")
        print(classification_report(y_test, pred, digits=4, target_names=["负面(0)", "正面(1)"]))

        prfs = precision_recall_fscore_support(y_test, pred, labels=[1,0], zero_division=0)
        for idx, label in enumerate([1,0]):
            name = "正面" if label == 1 else "负面"
            precision = prfs[0][idx]
            recall = prfs[1][idx]
            f1 = prfs[2][idx]
            support = prfs[3][idx]
            print(f"{name} (label={label}) - support={support}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    joblib.dump(models.get('en'), "w2v_en.pkl")
    joblib.dump(models.get('zh'), "w2v_zh.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(svm_model, "svm_model.pkl")
    print("模型已保存: w2v_en.pkl, w2v_zh.pkl, scaler.pkl, svm_model.pkl")
