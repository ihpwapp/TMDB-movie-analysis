WITH director_ids AS (
    SELECT movie_id, array_agg(person_id ORDER BY person_id) AS ids
    FROM movie_crew
    WHERE job = 'Director'
    GROUP BY movie_id
),
company_ids AS (
    SELECT movie_id, array_agg(company_id ORDER BY company_id) AS ids
    FROM movie_companies
    GROUP BY movie_id
),
cast_ids AS (
    SELECT movie_id, array_agg(person_id ORDER BY rn) AS ids
    FROM (
        SELECT movie_id, person_id, ROW_NUMBER() OVER (PARTITION BY movie_id ORDER BY cast_order ASC) AS rn
        FROM movie_cast
    ) cr
    WHERE cr.rn <= 3
    GROUP BY movie_id
)
SELECT m.movie_id, m.release_date, m.budget, m.revenue, m.belongs_to_collection_id, d.ids AS director_ids, c.ids AS company_ids, ca.ids AS cast_ids
FROM movies m
LEFT JOIN director_ids d ON m.movie_id = d.movie_id
LEFT JOIN company_ids c ON m.movie_id = c.movie_id
LEFT JOIN cast_ids ca ON m.movie_id = ca.movie_id;