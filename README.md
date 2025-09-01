# Finanzas


Esta aplicación de Streamlit permite analizar y visualizar datos financieros.

## Aplicación en vivo
Puedes acceder a la aplicación en Streamlit aquí: [Ver aplicación](https://economia.streamlit.app/)


## Funcionalidades

- Ingresar tickers de acciones y un benchmark.
- Construcción de un portafolio con los tickers introducidos.
' Comparación contra Benchmark.
- Análisis sectorial del portafolio.
- Visualizar gráficos de rendimiento y riesgo.
- Construcción de la Frontera Eficiente de Markowitz.
- CAPM de Sharpe y Modelo de Factores Fama French.

## Tecnologías

- Python 3.x
- Streamlit
- pandas, numpy, matplotlib, seaborn
- statsmodels
- quantstats
- yfinance, requests

## Cómo ejecutar localmente

1. Clonar el repositorio:

```bash
git clone https://github.com/rojasm22/finanzas.git
cd finanzas
```
 
## Construcción y Análisis

Para este análisis se utilizaron precios ajustados de Yahoo Finance correspondientes a los siguientes tickers: 

**CTVA,VMC,DD,NUE,PPG,DASH,RBLX,EA,LYV,TKO,CMG,ROST,F,YUM,EBAY,MNST,KMB,KR,TGT,SYY,EOG,MPC,PSX,OXY,VLO,PNC,COIN,BK,TRV,HOOD,COR,BDX,EW,CAH,A,EMR,CSX,FDX,URI,CMI,SPG,PSA,CCI,AVB,IRM,MSI,NET,FICO,TTD,HUBS,D,EXC,ED,NRG,AWK**.

Utilizando el **S&P 500 (^GSPC)** como benchmark de referencia.  

A continuación, se presentan los principales pasos del análisis y sus resultados visuales:

### Índice VW vs Benchmark

En primer lugar, se calcula el **Valor de Mercado (MVE)**:

$$
\text{MVE}_{jt} = \frac{P_{jt}}{\dot{N}_j}
$$




