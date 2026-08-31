SELECT 
    -- Core Identifiers & Metadata
    m.movie_id,
    m.imdb_id,
    m.title,
    m.original_title,
    m.status,
    m.original_language,
    m.adult,
    m.video,
    
    -- Temporal Features
    m.release_date,
    EXTRACT(YEAR FROM m.release_date) AS release_year,
    EXTRACT(MONTH FROM m.release_date) AS release_month,
    EXTRACT(DOW FROM m.release_date) AS release_day_of_week,
    
    -- Continuous Quantitative Metrics (Financials, Audience Scores, Runtime)
    m.budget,
    m.revenue,
    (m.revenue - m.budget) AS net_profit,
    CASE 
        WHEN m.budget > 0 THEN (m.revenue / CAST(m.budget AS DECIMAL))
        ELSE NULL 
    END AS roi,
    m.runtime,
    m.popularity AS movie_popularity,
    m.vote_average,
    m.vote_count,
    
    -- Franchise / Collection Metadata
    c.name AS collection_name,
    CASE WHEN m.belongs_to_collection_id IS NOT NULL THEN 1 ELSE 0 END AS is_part_of_franchise,
    
    -- Primary Category Aggregations
    pg.primary_genre,
    pc.primary_production_company,
    pco.primary_country,
    
    -- Primary Director & Key Cast Metadata
    dir.director_name,
    cast_agg.lead_cast,
    
    -- Timestamps
    m.created_at,
    m.updated_at

FROM movies m

-- Join Collection info
LEFT JOIN collections c 
    ON m.belongs_to_collection_id = c.collection_id

-- Subquery: Extract Primary Genre per movie
LEFT JOIN (
    SELECT 
        mg.movie_id,
        g.name AS primary_genre
    FROM (
        SELECT 
            movie_id, 
            genre_id,
            ROW_NUMBER() OVER (PARTITION BY movie_id ORDER BY genre_id) AS rn
        FROM movie_genres
    ) mg
    JOIN genres g ON mg.genre_id = g.genre_id
    WHERE mg.rn = 1
) pg ON m.movie_id = pg.movie_id

-- Subquery: Extract Primary Production Company per movie
LEFT JOIN (
    SELECT 
        mc.movie_id,
        comp.name AS primary_production_company
    FROM (
        SELECT 
            movie_id, 
            company_id,
            ROW_NUMBER() OVER (PARTITION BY movie_id ORDER BY company_id) AS rn
        FROM movie_companies
    ) mc
    JOIN production_companies comp ON mc.company_id = comp.company_id
    WHERE mc.rn = 1
) pc ON m.movie_id = pc.movie_id

-- Subquery: Extract Primary Production Country per movie
LEFT JOIN (
    SELECT 
        prc.movie_id,
        cou.english_name AS primary_country
    FROM (
        SELECT 
            movie_id, 
            iso_3166_1,
            ROW_NUMBER() OVER (PARTITION BY movie_id ORDER BY iso_3166_1) AS rn
        FROM production_countries
    ) prc
    JOIN countries cou ON prc.iso_3166_1 = cou.iso_3166_1
    WHERE prc.rn = 1
) pco ON m.movie_id = pco.movie_id

-- Subquery: Extract Director
LEFT JOIN (
    SELECT 
        mc.movie_id,
        STRING_AGG(p.name, ', ') AS director_name
    FROM movie_crew mc
    JOIN people p ON mc.person_id = p.person_id
    WHERE mc.job = 'Director'
    GROUP BY mc.movie_id
) dir ON m.movie_id = dir.movie_id

-- Subquery: Aggregate Top 3 Lead Actors into a single string (PostgreSQL syntax)
LEFT JOIN (
    SELECT 
        cast_rank.movie_id,
        STRING_AGG(p.name, ', ') AS lead_cast
    FROM (
        SELECT 
            movie_id, 
            person_id, 
            cast_order,
            ROW_NUMBER() OVER (PARTITION BY movie_id ORDER BY cast_order ASC) AS rn
        FROM movie_cast
    ) cast_rank
    JOIN people p ON cast_rank.person_id = p.person_id
    WHERE cast_rank.rn <= 3
    GROUP BY cast_rank.movie_id
) cast_agg ON m.movie_id = cast_agg.movie_id

-- Filter out invalid movies
WHERE m.budget > 500000
  AND m.revenue >= (m.budget / 10.0) AND m.runtime >= 30;