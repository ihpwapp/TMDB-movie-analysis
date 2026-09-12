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



