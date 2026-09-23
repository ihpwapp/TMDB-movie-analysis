# TMDB-movie-analysis

questions to answer - is the movie gonna be successful commercially, which features correlates most with a movie success, How much can movie metadata predict revenue? 


benefactor for this - studios and producers and actor

decision supported - whether a movie should be greenlit

success? - a movie makes 2.5x its initial budget

assumptions and limitation to take account - marketing spent, inflation for older movies, indie movies blowing up, WOM


DATA 

movies from 1925 - 2027 (~15k)
people (~530k)
data heavily skewed on hence using log1p

data that is known when making the prediction - budget, release date, genre, company, cast, director, keywoards


start with baseline of = [
    budget
    runtime
    release_year
    release_month
    genre   
    country
    language

]

candidate_features = {

    # Movie
    "movie": [
        "budget",
        "runtime",
        "original_language",
        "status",
    ],

    # Release
    "release": [
        "release_year",
        "release_month",
        "release_quarter",
        "release_day_of_week",
    ],

    # Genre
    "genre": [
        "primary_genre",
        "genre_count",
        # individual genre flags
    ],

    # Production
    "production": [
        "primary_production_company",
        "primary_production_country",
        "production_company_count",
        "production_country_count",
    ],

    # Director
    "director": [
        "director_name",
        "director_movie_count",
        "director_previous_avg_revenue",
        "director_previous_avg_budget",
        "director_previous_avg_roi",
        "director_previous_max_revenue",
    ],

    # Cast
    "cast": [
        "cast_size",
        "cast_avg_popularity",
        "cast_max_popularity",
        "cast_previous_avg_revenue",
    ],

    # Crew
    "crew": [
        "crew_size",
        "producer_count",
        "writer_count",
        "composer_count",
    ],

    # Keywords
    "keywords": [
        "keyword_count",
        "is_superhero",
        "is_sequel",
        "is_remake",
        "is_based_on_novel",
        "is_based_on_true_story",
    ],

    # Franchise
    "franchise": [
        "is_part_of_collection",
        "franchise_movie_count",
        "franchise_previous_avg_revenue",
        "franchise_previous_avg_roi",
    ],

    # Certification
    "certification": [
        "certification",
    ],

    # TMDB engagement
    "engagement": [
        "popularity",
        "vote_average",
        "vote_count",
    ],
}