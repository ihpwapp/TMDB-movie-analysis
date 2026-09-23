WITH pg AS (
    SELECT mg.movie_id, g.name AS primary_genre
    FROM (
        SELECT movie_id, genre_id, ROW_NUMBER() OVER (PARTITION BY movie_id ORDER BY genre_id) AS rn
        FROM movie_genres
    ) mg
    JOIN genres g ON mg.genre_id = g.genre_id
    WHERE mg.rn = 1
),
pc AS (
    SELECT mc.movie_id, comp.name AS primary_production_company
    FROM (
        SELECT movie_id, company_id, ROW_NUMBER() OVER (PARTITION BY movie_id ORDER BY company_id) AS rn
        FROM movie_companies
    ) mc
    JOIN production_companies comp ON mc.company_id = comp.company_id
    WHERE mc.rn = 1
),
pco AS (
    SELECT prc.movie_id, cou.english_name AS primary_country
    FROM (
        SELECT movie_id, iso_3166_1, ROW_NUMBER() OVER (PARTITION BY movie_id ORDER BY iso_3166_1) AS rn
        FROM production_countries
    ) prc
    JOIN countries cou ON prc.iso_3166_1 = cou.iso_3166_1
    WHERE prc.rn = 1
),
dir AS (
    SELECT mc.movie_id, STRING_AGG(p.name, ', ') AS director_name
    FROM movie_crew mc
    JOIN people p ON mc.person_id = p.person_id
    WHERE mc.job = 'Director'
    GROUP BY mc.movie_id
),
cast_agg AS (
    SELECT cr.movie_id, STRING_AGG(p.name, ', ' ORDER BY cr.cast_order ASC) AS cast
    FROM (
        SELECT movie_id, person_id, cast_order, ROW_NUMBER() OVER (PARTITION BY movie_id ORDER BY cast_order ASC) AS rn
        FROM movie_cast
    ) cr
    JOIN people p ON cr.person_id = p.person_id
    GROUP BY cr.movie_id
),
genre_cnt AS (
    SELECT movie_id, COUNT(*) AS genre_count
    FROM movie_genres
    GROUP BY movie_id
),
company_cnt AS (
    SELECT movie_id, COUNT(*) AS production_company_count
    FROM movie_companies
    GROUP BY movie_id
),
country_cnt AS (
    SELECT movie_id, COUNT(*) AS production_country_count
    FROM production_countries
    GROUP BY movie_id
),
cast_cnt AS (
    SELECT movie_id, COUNT(*) AS cast_size
    FROM movie_cast
    GROUP BY movie_id
),
crew_cnt AS (
    SELECT movie_id, COUNT(*) AS crew_size
    FROM movie_crew
    GROUP BY movie_id
),
producer_cnt AS (
    SELECT movie_id, COUNT(*) AS producer_count
    FROM movie_crew
    WHERE job = 'Producer'
    GROUP BY movie_id
),
writer_cnt AS (
    SELECT movie_id, COUNT(*) AS writer_count
    FROM movie_crew
    WHERE job = 'Writer'
    GROUP BY movie_id
),
composer_cnt AS (
    SELECT movie_id, COUNT(*) AS composer_count
    FROM movie_crew
    WHERE job = 'Original Music Composer'
    GROUP BY movie_id
),
keyword_cnt AS (
    SELECT movie_id, COUNT(*) AS keyword_count
    FROM movie_keywords
    GROUP BY movie_id
)
SELECT m.movie_id, m.imdb_id, m.title, m.original_title, m.status, m.original_language, m.adult, m.video, m.release_date, EXTRACT(YEAR FROM m.release_date) AS release_year, EXTRACT(MONTH FROM m.release_date) AS release_month, EXTRACT(DOW FROM m.release_date) AS release_day_of_week, m.budget, m.revenue, (m.revenue - m.budget) AS net_profit, CASE WHEN m.budget > 0 THEN (m.revenue / CAST(m.budget AS DECIMAL)) ELSE NULL END AS roi, m.runtime, m.popularity AS movie_popularity, m.vote_average, m.vote_count, c.name AS collection_name, CASE WHEN m.belongs_to_collection_id IS NOT NULL THEN 1 ELSE 0 END AS is_part_of_franchise, pg.primary_genre, pc.primary_production_company, pco.primary_country, dir.director_name, cast_agg.cast, m.overview, m.tagline, COALESCE(gc.genre_count, 0) AS genre_count, COALESCE(cc.production_company_count, 0) AS production_company_count, COALESCE(pc2.production_country_count, 0) AS production_country_count, COALESCE(ca.cast_size, 0) AS cast_size, COALESCE(crew.crew_size, 0) AS crew_size, COALESCE(pro.producer_count, 0) AS producer_count, COALESCE(wr.writer_count, 0) AS writer_count, COALESCE(com.composer_count, 0) AS composer_count, COALESCE(kc.keyword_count, 0) AS keyword_count, EXISTS (SELECT 1 FROM movie_keywords mk JOIN keywords k ON mk.keyword_id = k.keyword_id WHERE mk.movie_id = m.movie_id AND (strpos(lower(k.name), 'superhero') > 0 OR strpos(lower(k.name), 'marvel cinematic universe') > 0 OR strpos(lower(k.name), 'dc extended universe') > 0 OR k.name = 'based on comic')) AS is_superhero, EXISTS (SELECT 1 FROM movie_keywords mk JOIN keywords k ON mk.keyword_id = k.keyword_id WHERE mk.movie_id = m.movie_id AND strpos(lower(k.name), 'sequel') > 0) AS is_sequel, EXISTS (SELECT 1 FROM movie_keywords mk JOIN keywords k ON mk.keyword_id = k.keyword_id WHERE mk.movie_id = m.movie_id AND strpos(lower(k.name), 'remake') > 0) AS is_remake, EXISTS (SELECT 1 FROM movie_keywords mk JOIN keywords k ON mk.keyword_id = k.keyword_id WHERE mk.movie_id = m.movie_id AND (strpos(lower(k.name), 'novel') > 0 OR strpos(lower(k.name), 'based on book') > 0 OR strpos(lower(k.name), 'children''s book') > 0)) AS is_based_on_novel, EXISTS (SELECT 1 FROM movie_keywords mk JOIN keywords k ON mk.keyword_id = k.keyword_id WHERE mk.movie_id = m.movie_id AND (strpos(lower(k.name), 'true story') > 0 OR strpos(lower(k.name), 'biopic') > 0)) AS is_based_on_true_story
FROM movies m
LEFT JOIN collections c ON m.belongs_to_collection_id = c.collection_id
LEFT JOIN pg ON m.movie_id = pg.movie_id
LEFT JOIN pc ON m.movie_id = pc.movie_id
LEFT JOIN pco ON m.movie_id = pco.movie_id
LEFT JOIN dir ON m.movie_id = dir.movie_id
LEFT JOIN cast_agg ON m.movie_id = cast_agg.movie_id
LEFT JOIN genre_cnt gc ON m.movie_id = gc.movie_id
LEFT JOIN company_cnt cc ON m.movie_id = cc.movie_id
LEFT JOIN country_cnt pc2 ON m.movie_id = pc2.movie_id
LEFT JOIN cast_cnt ca ON m.movie_id = ca.movie_id
LEFT JOIN crew_cnt crew ON m.movie_id = crew.movie_id
LEFT JOIN producer_cnt pro ON m.movie_id = pro.movie_id
LEFT JOIN writer_cnt wr ON m.movie_id = wr.movie_id
LEFT JOIN composer_cnt com ON m.movie_id = com.movie_id
LEFT JOIN keyword_cnt kc ON m.movie_id = kc.movie_id
WHERE m.budget > 500000 AND m.revenue >= (m.budget / 10.0) AND m.runtime >= 30;