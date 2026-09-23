import sys
from pathlib import Path

# allow `streamlit run app/streamlit_app.py` to resolve the scripts package
if __package__ is None:
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from app.presets import FORM_FIELDS, PRESETS, estimate_preset_defaults
from scripts.models.predict import load_pipeline, predict_revenue
from scripts.models.request import build_input_features, humanize, INPUT_KEYS
from scripts.prepare_dataset import prepare_ml_dataset

GREENLIGHT_MULTIPLE = 2.5


def format_money(value):
    if value is None or (value != value):
        return "n/a"
    return f"${value / 1e6:,.0f}M"


@st.cache_resource(show_spinner=False)
def get_pipeline():
    return load_pipeline()


@st.cache_data(show_spinner=False)
def get_dataset():
    return prepare_ml_dataset()


def make_inputs_from_row(row):
    return {
        "budget": float(row["budget"]),
        "runtime": float(row["runtime"]),
        "release_date": pd.Timestamp(row["release_date"]),
        "original_language": row["original_language"],
        "primary_genre": row["primary_genre"],
        "primary_country": row["primary_country"],
        "company": row["primary_production_company"],
        "director": row["director_name"],
        "cast": row["cast"],
        "franchise": row.get("collection_name", "") if not pd.isna(row.get("collection_name", np.nan)) else "",
        "genre_count": int(row["genre_count"]),
        "production_company_count": int(row["production_company_count"]),
        "production_country_count": int(row["production_country_count"]),
        "cast_size": int(row["cast_size"]),
        "crew_size": int(row["crew_size"]),
        "producer_count": int(row["producer_count"]),
        "writer_count": int(row["writer_count"]),
        "composer_count": int(row["composer_count"]),
        "keyword_count": int(row["keyword_count"]),
        "is_superhero": bool(row["is_superhero"]),
        "is_sequel": bool(row["is_sequel"]),
        "is_remake": bool(row["is_remake"]),
        "is_based_on_novel": bool(row["is_based_on_novel"]),
        "is_based_on_true_story": bool(row["is_based_on_true_story"]),
    }


def render_prediction(predicted, budget, actual=None):
    st.markdown(f"### Predicted worldwide revenue: **{format_money(predicted)}**")
    if budget and budget > 0:
        multiple = predicted / budget
        ok = multiple >= GREENLIGHT_MULTIPLE
        st.markdown(
            f"Predicted **{multiple:.2f}× budget** — "
            + (f"clears the {GREENLIGHT_MULTIPLE:.1f}× greenlight hurdle 🟢" if ok else f"below the {GREENLIGHT_MULTIPLE:.1f}× greenlight hurdle 🔴"))
    if actual is not None and actual == actual:
        st.caption(f"Actual revenue in the dataset: {format_money(actual)}")


def show_shap(pipe, row):
    # Explain one prediction against a small background sample; NaN features get
    # the background column median so the waterfall stays readable. Renders a
    # waterfall, a top-driver bar chart, and the underlying contribution table.
    import matplotlib.pyplot as plt
    import shap

    from scripts.models.predict import build_request_matrix
    from scripts.models.request import format_feature_value

    X_req = build_request_matrix(pd.DataFrame([row]), pipe)
    cols = list(X_req.columns)

    bg_matrix = build_request_matrix(get_dataset().sample(64, random_state=7), pipe)
    medians = np.nanmedian(bg_matrix[cols].to_numpy(dtype=float), axis=0)

    filled = X_req[cols].to_numpy(dtype=float).copy()
    filled = np.where(np.isnan(filled), medians, filled)

    explainer = shap.TreeExplainer(pipe["model"])
    sv = explainer.shap_values(pd.DataFrame(filled, columns=cols))
    exp = shap.Explanation(
        values=sv[0],
        base_values=float(explainer.expected_value),
        data=filled[0],
        feature_names=[humanize(c) for c in cols],
    )

    w1, w2 = st.columns(2)
    with w1:
        st.markdown("**Waterfall** — how drivers move the estimate")
        shap.plots.waterfall(exp, show=False, max_display=10)
        st.pyplot(plt.gcf())
    with w2:
        st.markdown("**Top drivers** (magnitude of effect)")
        shap.plots.bar(exp, show=False, max_display=10)
        st.pyplot(plt.gcf())
    plt.close("all")

    contrib = []
    for i, col in enumerate(cols):
        sv_value = float(exp.values[i])
        if sv_value != sv_value:
            continue
        raw_value = row.get(col) if col in row.index else filled[0, i]
        dollar_move = np.expm1(exp.base_values + sv_value) - np.expm1(exp.base_values)
        contrib.append({
            "Driver": humanize(col),
            "Value": format_feature_value(col, raw_value),
            "Impact (log-rev)": round(sv_value, 3),
            "Approx $ move": f"+${dollar_move:,.0f}" if dollar_move >= 0 else f"-${abs(dollar_move):,.0f}",
        })
    with st.expander("Full SHAP contribution table", expanded=False):
        table = pd.DataFrame(contrib).sort_values("Impact (log-rev)", key=abs, ascending=False)
        st.dataframe(table.head(20), width="stretch", hide_index=True)
        st.caption("'Approx $ move' is the change in the predicted revenue if only that feature changed (rough approximation).")


def convert_date(date_input):
    return pd.Timestamp(date_input.strftime("%Y-%m-%d"))


def opt_index(options, value):
    return list(options).index(value)


def apply_preset_state(ds, preset_key):
    keys = [f"ff_{f}" for f in FORM_FIELDS]
    applied = st.session_state.get("ff_applied_preset")
    if applied == preset_key and all(k in st.session_state for k in keys):
        return
    for name, value in estimate_preset_defaults(ds, preset_key).items():
        st.session_state[f"ff_{name}"] = value
    st.session_state["ff_applied_preset"] = preset_key


def seed_defaults(ds, preset_label):
    label_to_key = {label: key for label, key in PRESETS}
    apply_preset_state(ds, label_to_key.get(preset_label, None))


st.set_page_config(page_title="TMDB Hollywood ROI", page_icon="🎬", layout="wide")
st.title("🎬 TMDB Hollywood ROI — Revenue Predictor")
st.caption(
    "Predicts **worldwide box-office revenue** from pre-release information only "
    "(budget, runtime, release window, genre, production company, director, cast, "
    "franchise, keywords). Historical track-record features are leakage-safe: they use "
    "films released strictly before the movie."
)

with st.sidebar:
    st.subheader("About")
    st.markdown(
        f"- Best model: **XGBoost** on `historical_lead` -> **test R² 0.59** (2022+, held out).\n"
        f"- Greenlight hurdle used here: **≥ {GREENLIGHT_MULTIPLE:.1f}× budget**."
    )
    st.info(
        "The free-form tab approximates track records by name (from the reference "
        "filmography); the lookup tab uses exact id-based features. Predictions lean "
        "optimistic because the dataset undersamples flops."
    )

tabs = st.tabs(["Movie Lookup & Edit", "Greenlight Form"])

pipe = get_pipeline()
ds = get_dataset()

popular = ds.sort_values("revenue", ascending=False)
global_genres = sorted(ds["primary_genre"].dropna().unique())
global_languages = sorted(ds["original_language"].dropna().unique())
global_countries = sorted(ds["primary_country"].dropna().unique())

with tabs[0]:
    st.subheader("Pick an existing movie, tweak it, and see the drivers")

    options = popular["title"].astype(str) + f"  ({popular['release_year'].astype(int).astype(str)})"

    c_pick, c_clear = st.columns([0.8, 0.2])
    with c_pick:
        pick = st.selectbox("Movie", options, index=min(20, len(popular) - 1))
    with c_clear:
        st.write("")
        if st.button("🔄 Clear results", key="clear_results", use_container_width=True):
            st.session_state.pop("edit_result", None)

    row0 = popular.iloc[opt_index(options, pick)]

    with st.form("edit_form"):
        c1, c2, c3 = st.columns(3)
        budget = c1.number_input("Budget ($)", min_value=0, step=1_000_000, value=int(row0["budget"]))
        runtime = c2.number_input("Runtime (min)", min_value=1, value=int(row0["runtime"]))
        release = c3.date_input("Release date", value=pd.Timestamp(row0["release_date"]).date())
        c4, c5, c6 = st.columns(3)
        genre = c4.selectbox(
            "Primary genre", global_genres,
            index=opt_index(global_genres, row0["primary_genre"]),
        )
        language = c5.selectbox(
            "Original language", global_languages,
            index=opt_index(global_languages, row0["original_language"]),
        )
        country = c6.selectbox(
            "Country", global_countries,
            index=opt_index(global_countries, row0["primary_country"]),
        )
        c7, c8 = st.columns(2)
        company = c7.text_input("Production company", value=row0["primary_production_company"])
        director = c8.text_input("Director", value=row0["director_name"])
        submitted = st.form_submit_button("Predict & explain")

    stored = st.session_state.get("edit_result")

    if submitted:
        inputs = make_inputs_from_row(row0)
        inputs.update({
            "budget": float(budget),
            "runtime": float(runtime),
            "release_date": convert_date(release),
            "primary_genre": genre,
            "original_language": language,
            "primary_country": country,
            "company": company,
            "director": director,
        })
        row = build_input_features(inputs, pipe["name_stats"], orig_row=row0)
        predicted = float(predict_revenue(row, pipe)[0])
        st.session_state["edit_result"] = {
            "pick": pick,
            "title": row0["title"],
            "row": row,
            "predicted": predicted,
            "budget": float(budget),
            "actual": float(row0["revenue"]),
        }
        stored = st.session_state["edit_result"]

    if stored is not None and stored.get("pick") == pick:
        predicted = stored["predicted"]
        with st.expander(f"Selected: {stored['title']}", expanded=False):
            c_m, c_g, c_c = st.columns(3)
            c_m.metric("Budget", format_money(stored["budget"]))
            c_g.metric("Primary genre", row0["primary_genre"])
            c_c.metric("Franchise", row0["collection_name"] if pd.notna(row0["collection_name"]) else "—")
        render_prediction(predicted, stored["budget"], actual=stored["actual"])
        if predicted == predicted:
            show_shap(pipe, stored["row"])
    else:
        st.info("No prediction for this movie yet — press **Predict & explain**.")

with tabs[1]:
    st.subheader("Greenlight form — describe a not-yet-released movie")
    st.caption(
        "Pick a template to auto-fill the form from this dataset's typical stats for that "
        "category (director/company/star/franchise default to the most common names in the "
        "data). Everything is editable — leaving a name empty means 'no track record'."
    )

    preset_label = st.segmented_control(
        "Template (auto-fills the form)",
        options=[label for label, _ in PRESETS],
        default="Blank (dataset median)",
        key="ff_preset_control",
    )
    seed_defaults(ds, preset_label)

    with st.form("free_form"):
        fa, fb, fc = st.columns(3)
        budget = fa.number_input("Budget ($)", min_value=0, step=1_000_000, key="ff_budget")
        runtime = fb.number_input("Runtime (min)", min_value=1, key="ff_runtime")
        release = fc.date_input("Release date", key="ff_release_date")

        g1, g2, g3 = st.columns(3)
        genre = g1.selectbox("Primary genre", global_genres, key="ff_primary_genre")
        language = g2.selectbox("Original language", global_languages, key="ff_original_language")
        country = g3.selectbox("Country", global_countries, key="ff_primary_country")

        h1, h2, h3 = st.columns(3)
        company = h1.text_input("Production company", key="ff_company",
                                placeholder="e.g. Universal Pictures")
        director = h2.text_input("Director", key="ff_director",
                                 placeholder="e.g. Christopher Nolan")
        franchise = h3.text_input("Franchise / collection", key="ff_franchise",
                                  placeholder="e.g. Jurassic Park Collection")
        cast = st.text_input("Billed cast (first = lead, comma-separated)", key="ff_cast",
                             placeholder="e.g. Zendaya, Timothee Chalamet, Florence Pugh")

        i1, i2, i3, i4, i5 = st.columns(5)
        cast_size = i1.number_input("Cast size", min_value=0, key="ff_cast_size")
        crew_size = i2.number_input("Crew size", min_value=0, key="ff_crew_size")
        producers = i3.number_input("Producers", min_value=0, key="ff_producer_count")
        writers = i4.number_input("Writers", min_value=0, key="ff_writer_count")
        composers = i5.number_input("Composers", min_value=0, key="ff_composer_count")

        j1, j2, j3, j4, j5, j6 = st.columns(6)
        super_ = j1.checkbox("Superhero", key="ff_is_superhero")
        sequel = j2.checkbox("Sequel", key="ff_is_sequel")
        remake = j3.checkbox("Remake", key="ff_is_remake")
        novel = j4.checkbox("Based on novel", key="ff_is_based_on_novel")
        true_story = j5.checkbox("True story", key="ff_is_based_on_true_story")
        keywords = j6.number_input("Keywords", min_value=0, key="ff_keyword_count")

        est = st.form_submit_button("Estimate revenue")

    if est:
        inputs = {
            "budget": float(budget),
            "runtime": float(runtime),
            "release_date": convert_date(release),
            "original_language": language,
            "primary_genre": genre,
            "primary_country": country,
            "company": company or "",
            "director": director or "",
            "cast": cast or "",
            "franchise": franchise or "",
            "genre_count": 1,
            "production_company_count": 1,
            "production_country_count": 1,
            "cast_size": int(cast_size),
            "crew_size": int(crew_size),
            "producer_count": int(producers),
            "writer_count": int(writers),
            "composer_count": int(composers),
            "keyword_count": int(keywords),
            "is_superhero": bool(super_),
            "is_sequel": bool(sequel),
            "is_remake": bool(remake),
            "is_based_on_novel": bool(novel),
            "is_based_on_true_story": bool(true_story),
        }
        row = build_input_features(inputs, pipe["name_stats"])
        predicted = float(predict_revenue(row, pipe)[0])
        predicted = predicted if predicted == predicted else np.nan
        render_prediction(predicted, float(budget))
        if predicted == predicted:
            show_shap(pipe, row)