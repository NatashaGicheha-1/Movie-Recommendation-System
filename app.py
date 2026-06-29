from html import escape
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path("MovieLens_Dataset") / "movies.csv"
LINKS_PATH = Path("MovieLens_Dataset") / "links.csv"
POSTER_CDN = "https://images.metahub.space/poster/medium/{imdb_id}/img"

GENRE_COLORS = {
    "Action": ("cf2e2e", "fff5f5"),
    "Adventure": ("d97706", "fff7ed"),
    "Animation": ("7c3aed", "faf5ff"),
    "Children": ("0891b2", "ecfeff"),
    "Comedy": ("ca8a04", "fefce8"),
    "Crime": ("334155", "f8fafc"),
    "Documentary": ("15803d", "f0fdf4"),
    "Drama": ("be123c", "fff1f2"),
    "Fantasy": ("6d28d9", "f5f3ff"),
    "Film-Noir": ("111827", "f9fafb"),
    "Horror": ("991b1b", "fef2f2"),
    "IMAX": ("0369a1", "f0f9ff"),
    "Musical": ("c026d3", "fdf4ff"),
    "Mystery": ("4338ca", "eef2ff"),
    "Romance": ("db2777", "fdf2f8"),
    "Sci-Fi": ("0f766e", "f0fdfa"),
    "Thriller": ("b45309", "fffbeb"),
    "War": ("57534e", "fafaf9"),
    "Western": ("92400e", "fff7ed"),
    "(no genres listed)": ("475569", "f8fafc"),
}


st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide",
)


@st.cache_data
def load_movies() -> pd.DataFrame:
    movies = pd.read_csv(DATA_PATH)
    if LINKS_PATH.exists():
        links = pd.read_csv(LINKS_PATH)
        movies = movies.merge(links[["movieId", "imdbId"]], on="movieId", how="left")
    else:
        movies["imdbId"] = pd.NA
    movies["imdb_id"] = movies["imdbId"].apply(
        lambda imdb_id: f"tt{int(imdb_id):07d}" if pd.notna(imdb_id) else None
    )
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    movies["genre_list"] = movies["genres"].str.split("|").apply(tuple)
    movies["clean_genres"] = movies["genre_list"].apply(lambda genres: ", ".join(genres))
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").fillna("Unknown")
    movies["display_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)
    movies["primary_genre"] = movies["genre_list"].str[0]
    return movies


@st.cache_data
def build_similarity(movies: pd.DataFrame) -> pd.DataFrame:
    genre_features = movies["genres"].str.get_dummies(sep="|")
    similarity = cosine_similarity(genre_features)
    return pd.DataFrame(
        similarity,
        index=movies["title"],
        columns=movies["title"],
    )


def recommend_movies(
    movie_title: str,
    movies: pd.DataFrame,
    similarity_df: pd.DataFrame,
    number_of_movies: int,
) -> pd.DataFrame:
    scores = similarity_df[movie_title].sort_values(ascending=False)
    scores = scores.drop(labels=[movie_title], errors="ignore").head(number_of_movies)

    recommendations = (
        movies.set_index("title")
        .loc[
            scores.index,
            [
                "movieId",
                "display_title",
                "year",
                "primary_genre",
                "clean_genres",
                "imdb_id",
            ],
        ]
        .reset_index()
    )
    recommendations.insert(1, "similarity", [round(score * 100, 1) for score in scores])
    return recommendations


def get_genre_names(movies: pd.DataFrame) -> list[str]:
    genres = sorted(
        {
            genre
            for genre_list in movies["genre_list"]
            for genre in genre_list
            if genre != "(no genres listed)"
        }
    )
    return genres


def placeholder_poster_url(title: str, year: str, genre: str) -> str:
    background, foreground = GENRE_COLORS.get(genre, ("1f2937", "ffffff"))
    poster_text = quote(f"{title}\n{year}")
    return f"https://placehold.co/420x620/{background}/{foreground}.png?text={poster_text}"


def poster_urls(title: str, year: str, genre: str, imdb_id: str | None) -> tuple[str, str]:
    fallback = placeholder_poster_url(title, year, genre)
    if imdb_id:
        return POSTER_CDN.format(imdb_id=imdb_id), fallback
    return fallback, fallback


def genre_chip_html(genre: str) -> str:
    background, _ = GENRE_COLORS.get(genre, ("475569", "f8fafc"))
    return f'<span class="genre-chip" style="--chip: #{background};">{escape(genre)}</span>'


def render_movie_card(row: pd.Series) -> None:
    imdb_id = row["imdb_id"] if pd.notna(row["imdb_id"]) else None
    poster_src, poster_fallback = poster_urls(
        str(row["display_title"]),
        str(row["year"]),
        str(row["primary_genre"]),
        imdb_id,
    )
    genres = [genre.strip() for genre in str(row["clean_genres"]).split(",")]
    chips = "".join(genre_chip_html(genre) for genre in genres[:3])

    st.markdown(
        f"""
        <div class="movie-card">
            <div class="poster-wrap">
                <img
                    class="poster-img"
                    src="{escape(poster_src)}"
                    alt="{escape(str(row['display_title']))} poster"
                    loading="lazy"
                    onerror="this.onerror=null;this.src='{escape(poster_fallback)}';"
                >
                <span class="match-badge">{row['similarity']}% match</span>
            </div>
            <div class="movie-name">{escape(str(row['display_title']))}</div>
            <div class="movie-year">{escape(str(row['year']))} · {escape(str(row['primary_genre']))}</div>
            <div class="chip-row">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


movies_df = load_movies()
movie_similarity_df = build_similarity(movies_df)
genre_names = get_genre_names(movies_df)
sorted_titles = movies_df["title"].sort_values().tolist()

st.markdown(
    """
    <style>
        :root {
            --bg: #06070a;
            --panel: #10141b;
            --panel-2: #151b24;
            --text: #f8fafc;
            --muted: #98a2b3;
            --line: rgba(255,255,255,.1);
            --accent: #e11d48;
            --accent-2: #38bdf8;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(225, 29, 72, .18), transparent 30rem),
                radial-gradient(circle at top right, rgba(56, 189, 248, .13), transparent 28rem),
                var(--bg);
            color: var(--text);
        }

        .main .block-container {
            max-width: 1220px;
            padding-top: 1.3rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, p, label, .stMarkdown {
            letter-spacing: 0;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSelectbox"] label,
        [data-testid="stSlider"] label,
        [data-testid="stMultiSelect"] label {
            color: #e5e7eb;
            font-weight: 650;
        }

        .stButton > button {
            width: 100%;
            min-height: 2.45rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,.16);
            background: rgba(16, 20, 27, .88);
            color: #f8fafc;
            font-weight: 750;
            transition: border-color .16s ease, background .16s ease, transform .16s ease;
        }

        .stButton > button:hover {
            border-color: rgba(225, 29, 72, .8);
            background: rgba(225, 29, 72, .18);
            color: #ffffff;
            transform: translateY(-1px);
        }

        .hero {
            min-height: 300px;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 2rem;
            background:
                linear-gradient(95deg, rgba(6, 7, 10, .98) 0%, rgba(6, 7, 10, .86) 48%, rgba(6, 7, 10, .35) 100%),
                url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=1600&q=80");
            background-size: cover;
            background-position: center;
            display: flex;
            align-items: end;
            margin-bottom: 1.2rem;
        }

        .hero-kicker {
            color: var(--accent-2);
            font-size: .8rem;
            text-transform: uppercase;
            font-weight: 800;
            letter-spacing: .08em;
            margin-bottom: .45rem;
        }

        .hero-title {
            font-size: clamp(2rem, 5vw, 4.5rem);
            line-height: .98;
            font-weight: 900;
            color: #ffffff;
            max-width: 780px;
            margin-bottom: .7rem;
        }

        .hero-copy {
            color: #cbd5e1;
            max-width: 700px;
            font-size: 1.03rem;
            line-height: 1.55;
        }

        .stats-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .75rem;
            margin: 1rem 0 1.1rem 0;
        }

        .stat-box {
            background: rgba(16, 20, 27, .82);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: .95rem 1rem;
        }

        .stat-label {
            color: var(--muted);
            font-size: .77rem;
            text-transform: uppercase;
            font-weight: 800;
            margin-bottom: .3rem;
        }

        .stat-value {
            color: var(--text);
            font-size: 1.35rem;
            font-weight: 850;
        }

        .section-title {
            color: #ffffff;
            font-size: 1.35rem;
            font-weight: 850;
            margin: 1.45rem 0 .8rem 0;
        }

        .genre-strip {
            display: flex;
            flex-wrap: wrap;
            gap: .5rem;
            margin: .6rem 0 1rem 0;
        }

        .genre-chip {
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            border: 1px solid color-mix(in srgb, var(--chip), white 22%);
            background: color-mix(in srgb, var(--chip), transparent 78%);
            color: #ffffff;
            border-radius: 999px;
            padding: .25rem .65rem;
            font-size: .8rem;
            font-weight: 750;
            margin: 0 .35rem .35rem 0;
            white-space: nowrap;
        }

        .selected-panel {
            display: grid;
            grid-template-columns: 140px 1fr;
            gap: 1rem;
            align-items: center;
            background: linear-gradient(135deg, rgba(21, 27, 36, .96), rgba(16, 20, 27, .9));
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            margin: .5rem 0 1rem 0;
        }

        .selected-panel img {
            width: 140px;
            aspect-ratio: 2 / 3;
            object-fit: cover;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,.12);
        }

        .selected-title {
            font-size: 1.5rem;
            color: #fff;
            font-weight: 850;
            margin-bottom: .25rem;
        }

        .selected-meta {
            color: var(--muted);
            margin-bottom: .7rem;
        }

        .movie-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 1rem;
        }

        .movie-card {
            min-width: 0;
        }

        .poster-wrap {
            position: relative;
            aspect-ratio: 2 / 3;
            border-radius: 8px;
            overflow: hidden;
            background: var(--panel-2);
            box-shadow: 0 16px 36px rgba(0,0,0,.35);
            border: 1px solid rgba(255,255,255,.1);
        }

        .poster-img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            transition: transform .18s ease, filter .18s ease;
        }

        .movie-card:hover .poster-img {
            transform: scale(1.035);
            filter: saturate(1.08);
        }

        .match-badge {
            position: absolute;
            left: .55rem;
            bottom: .55rem;
            background: rgba(0,0,0,.72);
            border: 1px solid rgba(255,255,255,.18);
            color: #ffffff;
            border-radius: 999px;
            padding: .25rem .55rem;
            font-size: .76rem;
            font-weight: 850;
            backdrop-filter: blur(8px);
        }

        .movie-name {
            color: #ffffff;
            font-weight: 800;
            line-height: 1.2;
            margin-top: .7rem;
            min-height: 2.4em;
            overflow-wrap: anywhere;
        }

        .movie-year {
            color: var(--muted);
            font-size: .86rem;
            margin: .22rem 0 .45rem 0;
        }

        .chip-row {
            min-height: 2.25rem;
        }

        .stAlert {
            background: rgba(21, 27, 36, .88);
            border: 1px solid var(--line);
            color: #e5e7eb;
        }

        @media (max-width: 980px) {
            .movie-grid {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
        }

        @media (max-width: 720px) {
            .hero {
                min-height: 260px;
                padding: 1.25rem;
            }

            .stats-row,
            .selected-panel {
                grid-template-columns: 1fr;
            }

            .selected-panel img {
                width: min(180px, 100%);
            }

            .movie-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: .85rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <div>
            <div class="hero-kicker">Content-based recommender</div>
            <div class="hero-title">Find your next movie by mood and genre.</div>
            <div class="hero-copy">
                Search for a movie you already like, explore the genres in the MovieLens dataset,
                and get similar recommendations in a visual movie-browsing layout.
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="stats-row">
        <div class="stat-box">
            <div class="stat-label">Movies</div>
            <div class="stat-value">{len(movies_df):,}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Genres</div>
            <div class="stat-value">{len(genre_names)}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Recommendation Type</div>
            <div class="stat-value">Similar Movies</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Browse Genres</div>', unsafe_allow_html=True)
genre_counts = movies_df.explode("genre_list")["genre_list"].value_counts()

if "active_genre" not in st.session_state:
    st.session_state.active_genre = "All"

genre_buttons = ["All"] + genre_names
for start in range(0, len(genre_buttons), 5):
    columns = st.columns(5)
    for column, genre in zip(columns, genre_buttons[start : start + 5]):
        label = genre
        if genre != "All":
            label = f"{genre} ({int(genre_counts.get(genre, 0))})"
        if column.button(label, key=f"genre_{genre}"):
            st.session_state.active_genre = genre

filtered_movies = movies_df
if st.session_state.active_genre != "All":
    filtered_movies = movies_df[
        movies_df["genre_list"].apply(
            lambda movie_genres: st.session_state.active_genre in movie_genres
        )
    ]

filtered_titles = filtered_movies["title"].sort_values().tolist()
if not filtered_titles:
    filtered_titles = sorted_titles

default_title = "Toy Story (1995)" if "Toy Story (1995)" in filtered_titles else filtered_titles[0]

st.caption(f"Active genre: {st.session_state.active_genre}")

control_left, control_right = st.columns([3, 1])

with control_left:
    selected_movie = st.selectbox(
        "Search movie, cinema, genre",
        options=filtered_titles,
        index=filtered_titles.index(default_title),
        help="Start typing to search through the movie titles.",
    )

with control_right:
    recommendation_count = st.slider(
        "Results",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

selected_row = movies_df.loc[movies_df["title"] == selected_movie].iloc[0]
selected_imdb_id = (
    selected_row["imdb_id"] if pd.notna(selected_row["imdb_id"]) else None
)
selected_poster, selected_poster_fallback = poster_urls(
    str(selected_row["display_title"]),
    str(selected_row["year"]),
    str(selected_row["primary_genre"]),
    selected_imdb_id,
)
selected_chips = "".join(
    genre_chip_html(genre.strip()) for genre in str(selected_row["clean_genres"]).split(",")
)

st.markdown(
    f"""
    <div class="selected-panel">
        <img
            src="{escape(selected_poster)}"
            alt="{escape(str(selected_row['display_title']))} poster"
            loading="lazy"
            onerror="this.onerror=null;this.src='{escape(selected_poster_fallback)}';"
        >
        <div>
            <div class="selected-title">{escape(str(selected_row["display_title"]))}</div>
            <div class="selected-meta">{escape(str(selected_row["year"]))} / Movie ID {int(selected_row["movieId"])}</div>
            <div>{selected_chips}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

recommendations_df = recommend_movies(
    selected_movie,
    movies_df,
    movie_similarity_df,
    recommendation_count,
)

st.markdown('<div class="section-title">Recommended For You</div>', unsafe_allow_html=True)

recommendation_rows = list(recommendations_df.iterrows())
for start in range(0, len(recommendation_rows), 5):
    columns = st.columns(5)
    for column, (_, row) in zip(columns, recommendation_rows[start : start + 5]):
        with column:
            render_movie_card(row)

with st.expander("View recommendations as a table"):
    st.dataframe(
        recommendations_df.rename(
            columns={
                "title": "Movie",
                "movieId": "Movie ID",
                "similarity": "Similarity %",
                "display_title": "Title",
                "year": "Year",
                "primary_genre": "Main Genre",
                "clean_genres": "Genres",
            }
        ),
        hide_index=True,
        width="stretch",
    )
