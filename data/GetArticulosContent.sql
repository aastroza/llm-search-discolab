WITH ArticuloContent AS (
    SELECT 
        [ID_ARTICULO],
        STRING_AGG(CAST([CONTENIDO_PARRAFO] AS VARCHAR(MAX)), '\n') AS CONTENIDO_ARTICULO -- Convert text to varchar
    FROM 
        PARRAFO
    GROUP BY 
        [ID_ARTICULO]
)
SELECT 
    c.[ID_CONSTITUCION],
    c.[NOMBRE] AS NOMBRE_CONSTITUCION,
    cap.[ID_CAPITULO],
    cap.[NOMBRE] AS NOMBRE_CAPITULO,
    cap.[TITULO] AS TITULO_CAPITULO,
    a.[ID_ARTICULO],
    a.[NOMBRE] AS NOMBRE_ARTICULO,
    ac.CONTENIDO_ARTICULO
FROM 
    CONSTITUCION c
INNER JOIN 
    CAPITULO cap ON c.[ID_CONSTITUCION] = cap.[ID_CONSTITUCION]
INNER JOIN 
    ARTICULO a ON cap.[ID_CAPITULO] = a.[ID_CAPITULO]
INNER JOIN 
    ArticuloContent ac ON a.[ID_ARTICULO] = ac.[ID_ARTICULO]