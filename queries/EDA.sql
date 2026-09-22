-- seasonality performance
SELECT 
    EXTRACT(MONTH FROM release_date) AS release_month,
    TO_CHAR(release_date, 'Month') AS month_name,
    EXTRACT(DOW FROM release_date) AS day_of_week_num,
    TO_CHAR(release_date, 'Day') AS day_of_week_name,
    COUNT(movie_id) AS total_movies,
    SUM(budget) AS total_budget,
    SUM(revenue) AS total_revenue,
    AVG(revenue) AS avg_revenue,
    AVG(CASE WHEN budget > 0 THEN (revenue / CAST(budget AS DECIMAL)) ELSE NULL END) AS avg_roi,
    AVG(vote_average) AS avg_rating
FROM movies
WHERE release_date IS NOT NULL
GROUP BY 
    EXTRACT(MONTH FROM release_date),
    TO_CHAR(release_date, 'Month'),
    EXTRACT(DOW FROM release_date),
    TO_CHAR(release_date, 'Day')
ORDER BY avg_revenue;



-- movie count by year
SELECT 
    EXTRACT(YEAR FROM release_date) AS release_year,
    COUNT(movie_id) AS total_movies,
    SUM(budget) AS total_budget,
    SUM(revenue) AS total_revenue,
    AVG(revenue) AS avg_revenue,
    AVG(CASE WHEN budget > 0 THEN (revenue / CAST(budget AS DECIMAL)) ELSE NULL END) AS avg_roi,
    AVG(vote_average) AS avg_rating
FROM movies
WHERE release_date IS NOT NULL
GROUP BY EXTRACT(YEAR FROM release_date)
ORDER BY release_year;   


-- movie count by language 
select 
	count(m.movie_id) as total_movies,
	m.original_language 
from movies m
group by m.original_language 
order by count(m.movie_id ) DESC;


--movie count by genre
select 
	count(m.movie_id) as total_movies,
	g.name
from movies m 
left join movie_genres mg on mg.movie_id = m.movie_id
left join genres g on mg.genre_id = g.genre_id
group by g."name" 
order by count(m.movie_id) DESC;


-- Budget vs revenue
SELECT 
    title,
    budget,
    revenue,
    (revenue - budget) AS net_profit,
    ROUND(((revenue - budget) / budget) * 100, 2) AS roi_percentage
FROM movies
WHERE budget > 0 AND revenue > 0
ORDER BY net_profit DESC;

-- release year vs revenue
select 
	extract(year from release_date) as release_year,
    SUM(budget) as budget,
    SUM(revenue) as revenue,
    SUM(revenue - budget) as net_profit,
    ROUND(((SUM(revenue) - SUM(budget)) / SUM(budget)) * 100, 2) as roi_percentage
from movies
where budget > 0 and revenue > 0 
group by extract(year from release_date)
order by release_year desc;

