# Construcción y Análisis

Para este análisis se utilizaron precios ajustados de Yahoo Finance correspondientes a los siguientes tickers:  

**CTVA, VMC, DD, NUE, PPG, DASH, RBLX, EA, LYV, TKO, CMG, ROST, F, YUM, EBAY, MNST, KMB, KR, TGT, SYY, EOG, MPC, PSX, OXY, VLO, PNC, COIN, BK, TRV, HOOD, COR, BDX, EW, CAH, A, EMR, CSX, FDX, URI, CMI, SPG, PSA, CCI, AVB, IRM, MSI, NET, FICO, TTD, HUBS, D, EXC, ED, NRG, AWK**  

Se utilizó el **S&P 500 (^GSPC)** como benchmark de referencia.

El análisis comprende desde 01-1998 hasta 12-2024
---

## Índice VW vs Benchmark

En primer lugar, se calcula la **capitalización de mercado (MVE)** de cada empresa \(j\) en el período \(t\):

$$
MVE_{jt} = P_{jt} \cdot N_j
$$

donde:  
- $MVE_{jt}$ representa la capitalización de mercado de la empresa j en el período t.  
- $P_{jt}$ es el precio ajustado de la acción j en el período t.  
- $N_j$ es la cantidad de acciones en circulación.



Seguidamente, se calcula el **peso de cada empresa** en el índice ponderado por capitalización de mercado:

$$
w_{t-1} = \frac{MVE_{jt}}{\sum_{j=1}^{M_{t-1}} MVE_{jt}}
$$

Con base en estas ponderaciones, se calculan los **retornos del mercado**:

$$
r_{mt} = \sum_{j=1}^{M_{t-1}} w_{t-1} \, r_{jt}
$$

Finalmente, el **índice value-weighted (VW)** se calcula como el producto acumulado de los retornos del mercado:

$$
vw_t = \kappa \prod_{\tau=1}^{t} (1 + r_{m\tau})
$$

![Comparación del S&P500 con el índice ponderado por capitalización de mercado](imagenes/1.png)

 Al analizar el comportamiento del índice S&P 500 frente al índice propio, se observa que ambos presentan una tendencia creciente similar. Sin embargo, el índice propio parece ser más vulnerable a shocks económicos y a presentar un mayor crecimiento. Esta diferencia se explica, en parte, por la composición de cada uno: el S&P 500 incluye las 500 empresas de mayor capitalización bursátil en EE. UU., mientras que el índice propio incorpora empresas de los 11 sectores económicos, incluyendo algunas que el S&P 500 considera demasiado pequeñas pero con altos rendimientos.

 ### Posibles sesgos del portafolio y del benchmark:
- Sesgo de supervivencia: El S&P 500 excluye empresas que quiebran o son absorbidas por
 otras. Al mostrar solo las que sobreviven y prosperan, se genera una imagen distorsionada del
 mercado, más optimista de lo que realmente es. Esto puede llevar a sobreestimar el rendimiento
 histórico del índice.
 - Sesgo de selección: Este sesgo proviene del proceso de inclusión de empresas en el índice.
 En el caso del S&P 500, solo se incluyen compañías que cumplen ciertos criterios de capital
ización, liquidez, rentabilidad y gobierno corporativo. Esto deja fuera a muchas empresas que,
 aunque relevantes, no califican, limitando la diversidad del índice y su capacidad de reflejar
 movimientos más amplios del mercado. Igualmente, el índice propio incurre en este sesgo,
 puesto que incluye empresas grandes dentro de cada sector.
 - Sesgo de capitalización: En el S&P 500, las empresas más grandes tienen un peso mucho
 mayor. Esto significa que los movimientos de unas pocas compañías (como Apple o Microsoft)
 pueden influir desproporcionadamente en el comportamiento del índice, ocultando la dinámica
 de cientos de empresas más pequeñas.

---

 ## Índice Herfindahl-Hirschman
 
 El índice Herfindahl-Hirschman (HHI) es una medida utilizada para evaluar el nivel de concetración en un mercado, por tanto, permite estimar el grado de poder de mercado que tienen las empresas participantes. Se calcula sumando los cuadrados de las cuotas de mercado de cada empresa.

$HHI \equiv h = \sum^{n}_{i=1}w_{i}^{2}$

![Evolución temporal del HHI](imagenes/2.png)

Esta figura muestra una tendencia decreciente en el índice Herfindahl-Hirschman (HHI), lo que evidencia una disminución progresiva en la concentración del mercado a lo largo del período analizado. Esto implica que el índice propio ha evolucionado hacia una estructura más equitativa, donde el peso ya no se concentra en unas pocas compañías dominantes. Este comportamiento sugiere un aumento en la diversidad y la competencia relativa dentro del índice, atribuible al ingreso paulatino de nuevas empresas. La expansión sectorial y la inclusión de firmas más pequeñas fortalecen su capacidad representativa del mercado en general.


![Variación interanual del HHI](imagenes/4.png)
 El análisis de la variación interanual afirma que, aunque la tendencia general del índice es hacia una menor concentración (como se ve en los niveles absolutos del HHI), existen momentos de reversión o choque en los que el poder de mercado se concentra temporalmente. Esto resalta la importancia de considerar tanto el nivel como la variación de la concentración al evaluar la representatividad y estabilidad del índice

 ![Número de activos efectivos del HHI](imagenes/3.png)

Por otro lado, resulta interesante que, hacia el final del período, la cantidad efectiva de compañías es 49.25 y un h= 0.0203, pese a que el índice incluye 55 en total. Esto indica que al menos seis empresas tienen una participación tan baja que su peso relativo en el índice es prácticamente nulo, lo cual refleja cierto grado de concentración residual, aunque en niveles bajos. Recordando que el número efetivo de activos es $n^{\star} = \frac{1}{h}$