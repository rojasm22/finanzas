# Construcción y Análisis

Para este análisis se utilizaron precios ajustados de Yahoo Finance correspondientes a los siguientes tickers:  

**CTVA, VMC, DD, NUE, PPG, DASH, RBLX, EA, LYV, TKO, CMG, ROST, F, YUM, EBAY, MNST, KMB, KR, TGT, SYY, EOG, MPC, PSX, OXY, VLO, PNC, COIN, BK, TRV, HOOD, COR, BDX, EW, CAH, A, EMR, CSX, FDX, URI, CMI, SPG, PSA, CCI, AVB, IRM, MSI, NET, FICO, TTD, HUBS, D, EXC, ED, NRG, AWK**  

Se utilizó el **S&P 500 (^GSPC)** como benchmark de referencia.

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

---

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

