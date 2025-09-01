## Construcción y Análisis

Para este análisis se utilizaron precios ajustados de Yahoo Finance correspondientes a los siguientes tickers: 

**CTVA,VMC,DD,NUE,PPG,DASH,RBLX,EA,LYV,TKO,CMG,ROST,F,YUM,EBAY,MNST,KMB,KR,TGT,SYY,EOG,MPC,PSX,OXY,VLO,PNC,COIN,BK,TRV,HOOD,COR,BDX,EW,CAH,A,EMR,CSX,FDX,URI,CMI,SPG,PSA,CCI,AVB,IRM,MSI,NET,FICO,TTD,HUBS,D,EXC,ED,NRG,AWK**.

Utilizando el **S&P 500 (^GSPC)** como benchmark de referencia.  

A continuación, se presentan los principales pasos del análisis y sus resultados visuales:

### Índice VW vs Benchmark

En primer lugar, se calcula:
$MVE_{jt} = P_{jt} \cdot N_{j}$

Donde $MVE_{jt}$, representa la capitalización de mercado de la empresa j en el período t, $P{jt} representa el precio ajustado de la acción j en el período t y $N{j}$ es la cantidad de acciones en circulación.

Seguidamente se cálcula el peso de cada empresa en el índice ponderado por capitalización de mercado, permitiendo que el número de empresas cambie entre períodos:
 
$ w_{t-1} = \frac{MVE_{jt}}{\Sum^{M_{t-1}}_{j = 1}MVE_{jt} } $

Con base en las ponderaciones se calculan los retornos del mercado:

$ r_{mt} = \Sum^{M_{t-1}}_{j = 1} w_{t-1} r_{jt} $

De esta forma el índice vw_{t} (value weighted) sería:

$vw_{t}= \kappa \Prod_{tau=1}^{t} 1+r_{mt}$