import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import calendar
import psutil
import quantstats as qs
import statsmodels.api as sm
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import plotting
import os
import itertools
from itertools import combinations
import requests, io, zipfile
# URLs de Fama-French
plt.style.use('https://github.com/kimichenn/nord-deep-mpl-stylesheet/raw/main/nord-deep.mplstyle')

# --- Clase principal ---
class PortafolioAnalyzer:
    def __init__(self, tickers, benchmark, start_date, end_date):
        """
        Inicializa el analizador de portafolios.
        
        Args:
            tickers: Lista de símbolos de acciones
            benchmark: Símbolo del índice de referencia
            start_date: Fecha de inicio del análisis
            end_date: Fecha final del análisis
        """
        # === PARÁMETROS DE ENTRADA ===
        self.tickers = tickers
        self.benchmark = benchmark
        self.start_date = start_date
        self.end_date = end_date
        
        # === DATOS BÁSICOS ===
        self.precios_ajustados = None
        self.shares_outstanding = None
        self.fama = None
        
        # === ANÁLISIS DE RETORNOS ===
        self.retornos_portafolio = None
        self.retornos_benchmark = None
        self.indice_benchmark = None
        self.indice_portafolio = None
        
        # === MÉTRICAS DE CONCENTRACIÓN ===
        self.vw = None  # Pesos por capitalización
        self.hhi = None  # Índice Herfindahl-Hirschman
        self.hhi_interanual = None
        self.activos_efectivos = None
        
        # === ANÁLISIS POR SECTORES ===
        self.sectores = None
        self.indice_sectores = None
        self.retornos_por_sector = None
        self.sectores_exceso = None
        
        # === OPTIMIZACIÓN DE PORTAFOLIOS ===
        # Matrices fundamentales de Markowitz
        self.A = None
        self.B = None 
        self.C = None
        self.D = None
        
        # Matrices auxiliares
        self.inv_cov = None
        self.V = None
        self.mu = None
        self.iota = None
        self.V1 = None
        
        # Cartera de mínima varianza
        self.media_mvp = None
        self.volatilidad_mvp = None



    def limpiar_datos(self):
        """Limpia todos los datos del analizador"""
        self.precios_ajustados = None
        self.shares_outstanding = None
        self.retornos_portafolio = None
        self.indice_benchmark = None
        self.indice_portafolio = None
        self.vw = None
        self.hhi = None
        self.hhi_interanual = None
        self.activos_efectivos = None
        self.sectores = None
        self.retornos_por_sector = None
        self.retornos_benchmark = None
        st.success("Todos los datos han sido limpiados")








    # --- Descarga de todos los datos ---
    def descargar_datos(self):
        total_tickers = self.tickers + [self.benchmark]
        data = yf.download(total_tickers, start=self.start_date, end=self.end_date, auto_adjust=True, interval='1mo')
        if len(total_tickers) == 1:
            self.precios_ajustados = pd.DataFrame({total_tickers[0]: data['Close']})
        else:
            self.precios_ajustados = data['Close']

        shares_data = {}
        sectores_data = {}
        for ticker in self.tickers:
            try:
                ticker_info = yf.Ticker(ticker)
                # Acciones en mercado de cada ticker
                shares = ticker_info.info.get('sharesOutstanding', 1)
                shares_data[ticker] = shares if shares and shares > 0 else 1

                # Sectores de cada acción
                sector = ticker_info.info.get('sector')
                sectores_data[ticker] = sector

            except Exception:
                shares_data[ticker] = 1
                sectores_data[ticker] = 'Desconocido'
        self.shares_outstanding = pd.Series(shares_data)
        self.sectores = pd.Series(sectores_data)
        



#### DESCARGAR DATOS DE FAMA FRENCH
    def descargar_factores_fama_french(self):
        """
        Descarga y une los factores Fama-French 3 factores y Momentum.
        Retorna un DataFrame mensual con las columnas:
        ['Mkt-RF','SMB','HML','RF','Mom']
        """

        urls = {
            "3_factors": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip",
            "momentum": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
        }

        factores = {}

        for key, url in urls.items():
            try:
                
                
                # Descargar ZIP con headers para evitar bloqueos
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                r = requests.get(url, headers=headers, timeout=30)
                r.raise_for_status()  # Lanza excepción si hay error HTTP
                
                # Abrir ZIP
                z = zipfile.ZipFile(io.BytesIO(r.content))
                
                # Buscar archivo CSV
                csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
                if not csv_files:
                    st.info(f"No se encontró archivo CSV en {key}")
                    continue
                csv_name = csv_files[0]
                
                # Leer con diferentes skiprows según el archivo
                if key == "3_factors":
                    skiprows = 3  # Ajustado para saltar headers correctamente
                else:  # momentum
                    skiprows = 13  # Ajustado para momentum
                
                # Leer CSV
                with z.open(csv_name) as csv_file:
                    df = pd.read_csv(csv_file, skiprows=skiprows)
                
                # Limpiar y procesar
                df = df.dropna(how='all')
                
                # Limpiar nombres de columnas
                df.columns = [str(c).strip() for c in df.columns]
                
                # La primera columna debería ser la fecha
                date_col = df.columns[0]
                df[date_col] = df[date_col].astype(str).str.strip()
                
                # Filtrar solo fechas válidas YYYYMM (6 dígitos)
                mask = df[date_col].str.match(r'^\d{6}$', na=False)
                df = df[mask].copy()
                
                if df.empty:
                    st.info(f"No se encontraron fechas válidas en {key}")
                    continue
                
                # Convertir fechas
                df['Date'] = pd.to_datetime(df[date_col], format='%Y%m').dt.to_period('M')
                df = df.set_index('Date')
                
                # Eliminar la columna original de fecha
                df = df.drop(columns=[date_col])
                
                # Convertir valores numéricos y dividir por 100
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 100
                
                # Eliminar filas que son completamente NaN después de la conversión
                df = df.dropna(how='all')
                
                factores[key] = df
                
            except Exception as e:
                st.info(f"Error descargando {key}: {str(e)}")
                factores[key] = pd.DataFrame()

        # Combinar los DataFrames
        df_factores_completos = pd.DataFrame()
        
        if factores.get('3_factors') is not None and not factores['3_factors'].empty:
            df_factores_completos = factores['3_factors'].copy()

        if factores.get('momentum') is not None and not factores['momentum'].empty:
            if df_factores_completos.empty:
                df_factores_completos = factores['momentum'].copy()
            else:
                df_factores_completos = df_factores_completos.join(factores['momentum'], how='outer')

            # Renombrar la última columna a 'Mom' si es momentum
            momentum_cols = [col for col in factores['momentum'].columns if 'mom' in col.lower()]
            if momentum_cols:
                old_name = momentum_cols[0]
                if old_name in df_factores_completos.columns:
                    df_factores_completos = df_factores_completos.rename(columns={old_name: 'Mom'})

        # Ordenar por fecha
        if not df_factores_completos.empty:
            df_factores_completos = df_factores_completos.sort_index()
        else:
            st.info("No se pudieron descargar los datos")

        self.fama = df_factores_completos
        return df_factores_completos
    




    def unir_factores_y_retornos(self):
        """
        Une los factores Fama-French con los retornos por sector.
        Filtra por fechas comunes y dentro del rango definido por start_date y end_date.
        Retorna un DataFrame combinado.
        """

        # Validaciones básicas
        if self.fama is None or self.fama.empty:
            raise ValueError("Los factores Fama-French no están disponibles.")
        if self.retornos_por_sector is None or self.retornos_por_sector.empty:
            raise ValueError("Los retornos por sector no están disponibles.")

        # Asegurar que los índices sean PeriodIndex mensual
        fama = self.fama.copy()
        sectores = self.retornos_por_sector.copy()

        if not isinstance(fama.index, pd.PeriodIndex):
            fama.index = pd.to_datetime(fama.index).to_period('M')
        if not isinstance(sectores.index, pd.PeriodIndex):
            sectores.index = pd.to_datetime(sectores.index).to_period('M')

        # Filtrar por fechas comunes
        fechas_comunes = fama.index.intersection(sectores.index)
        fechas_filtradas = fechas_comunes[(fechas_comunes >= pd.Period(self.start_date, freq='M')) &
                                        (fechas_comunes <= pd.Period(self.end_date, freq='M'))]

        fama_filtrado = fama.loc[fechas_filtradas]
        sectores_filtrado = sectores.loc[fechas_filtradas]


        # Restar RF promedio a todos los retornos de sectores
        sectores_exceso = sectores_filtrado.subtract(fama_filtrado['RF'], axis=0)

        # Unir horizontalmente
        df_combinado = pd.concat([sectores_exceso, fama_filtrado], axis=1)

        # Eliminar filas con NaNs completos
        df_combinado = df_combinado.dropna(how='all')
        df_combinado = df_combinado.iloc[1:]
        # Guardar resultado
        self.rf = fama_filtrado['RF']
        self.sectores_exceso = sectores_exceso
        self.df_combinado = df_combinado
        return df_combinado









   
    ## --- Regresiones estadisticamente significativas ---##
    
    def regresiones_ff(self):
        if not hasattr(self, 'df_combinado') or self.df_combinado is None:
            raise ValueError("Primero ejecuta unir_factores_y_retornos() para obtener df_combinado.")
        
        if self.sectores is None or not isinstance(self.sectores, pd.Series):
            raise ValueError("self.sectores debe ser una Serie con nombres de sectores.")
        
        factores = ['Mkt-RF', 'SMB', 'HML', 'Mom']
        sectores = self.sectores.unique()
        df_combinado = self.df_combinado
        resultados = {}

        for sector in sectores:
            if sector not in df_combinado.columns:
                continue

            y = df_combinado[sector].dropna()
            X_base = df_combinado.loc[y.index, factores]

            mejor_aic = float('inf')
            mejor_modelo = None
            mejor_subset = None

            # Probar todas las combinaciones posibles de factores (excepto vacío)
            for k in range(1, len(factores)+1):
                for subset in itertools.combinations(factores, k):
                    X_subset = X_base[list(subset)]
                    X_subset = sm.add_constant(X_subset)

                    # Alinear fechas válidas
                    valid_idx = X_subset.dropna().index.intersection(y.index)
                    X_valid = X_subset.loc[valid_idx]
                    y_valid = y.loc[valid_idx]

                    if len(y_valid) < len(subset) + 2:
                        continue  # Evitar modelos con muy pocos datos

                    try:
                        modelo = sm.OLS(y_valid, X_valid).fit()
                        if modelo.aic < mejor_aic:
                            mejor_aic = modelo.aic
                            mejor_modelo = modelo
                            mejor_subset = subset
                    except Exception:
                        continue  # Ignorar combinaciones que fallen

            if mejor_modelo:
                resultados[sector] = {
                    'factores_seleccionados': mejor_subset,
                    'modelo': mejor_modelo,  # Guardar el objeto modelo completo
                    'aic': mejor_aic,
                    'r_squared': mejor_modelo.rsquared,
                    'r_squared_adj': mejor_modelo.rsquared_adj,
                    'n_obs': int(mejor_modelo.nobs),
                    'resumen_texto': mejor_modelo.summary().as_text()  # Opcional
                }
            else:
                resultados[sector] = {
                    'error': "No se pudo ajustar ningún modelo válido."
                }

        return resultados


    def resumen_coeficientes(self):
        """
        Extrae solo los coeficientes de factores y alfa
        """
        resultados = self.regresiones_ff()
        resumen = []

        # Mapeo de nombres
        factor_mapping = {
            'Mkt-RF': 'Z_m', 
            'SMB': 'Z_s', 
            'HML': 'Z_v', 
            'Mom': 'Z_mom'
        }

        for sector, resultado in resultados.items():
            if isinstance(resultado, dict) and 'modelo' in resultado:
                modelo = resultado['modelo']
                factores_seleccionados = resultado['factores_seleccionados']
                
                # Inicializar con None todos los coeficientes
                coeficientes = {
                    'Alpha': None,
                    'Z_m': None, 
                    'Z_s': None, 
                    'Z_v': None, 
                    'Z_mom': None
                }
                
                # Extraer Alpha (intercepto)
                if 'const' in modelo.params:
                    coeficientes['Alpha'] = modelo.params['const']
                
                # Extraer coeficientes de factores seleccionados
                for factor in factores_seleccionados:
                    if factor in modelo.params:
                        col_name = factor_mapping[factor]
                        coeficientes[col_name] = modelo.params[factor]
                
                # Solo agregar sector y coeficientes
                fila = {'Sector': sector, **coeficientes}
                resumen.append(fila)
            
            else:
                # Sector con error
                resumen.append({
                    'Sector': sector,
                    'Alpha': None, 'Z_m': None, 'Z_s': None, 'Z_v': None, 'Z_mom': None
                })

        df_resumen = pd.DataFrame(resumen)
        df_resumen = df_resumen.set_index('Sector')
        
        # Redondear solo las columnas numéricas
        numeric_cols = df_resumen.select_dtypes(include=[np.number]).columns
        df_resumen[numeric_cols] = df_resumen[numeric_cols].round(4)
        
        self.df_resumen = df_resumen
        return df_resumen




















    # --- Cálculo de retornos ---
    def calcular_retornos(self):
        precios_portafolio = self.precios_ajustados[self.tickers]
        precios_benchmark = self.precios_ajustados[self.benchmark]

        # Alinear fechas
        precios_portafolio, precios_benchmark = precios_portafolio.align(
            precios_benchmark, join='inner', axis=0
        )

        # Calcular capitalización de mercado
        mve = precios_portafolio.multiply(self.shares_outstanding, axis=1)
        total_mve = mve.sum(axis=1).replace(0, np.nan)
        vw = mve.div(total_mve, axis=0).fillna(0)

        # Calcular retornos individuales (mantener NaN en primera fila)
        retornos_tickers = precios_portafolio.pct_change()
        retornos_benchmark = precios_benchmark.pct_change()

        # Alinear retornos con pesos
        retornos_tickers, retornos_benchmark = retornos_tickers.align(
            retornos_benchmark, join='inner', axis=0
        )
        vw_aligned = vw.reindex(retornos_tickers.index, method='ffill')

        # Calcular retorno ponderado del portafolio
        retornos_portafolio = (retornos_tickers * vw_aligned.shift(1)).sum(axis=1)

        # OBTENER LA FECHA INICIAL CORRECTA
        fecha_inicio = retornos_portafolio.index[0]  # Primera fecha disponible
        
        # Calcular índice empezando desde la fecha inicial con valor 100
        indice_portafolio = (1 + retornos_portafolio.loc[fecha_inicio:]).cumprod() * 100
        indice_benchmark = (1 + retornos_benchmark.loc[fecha_inicio:]).cumprod() * 100

        # Guardar resultados
        self.retornos_portafolio = retornos_portafolio.iloc[1:]  # Quitar solo el primer valor
        self.retornos_benchmark = retornos_benchmark.iloc[1:]
        self.indice_portafolio = indice_portafolio
        self.indice_benchmark = indice_benchmark
        self.vw = vw











    # --- Cálculo de HHI ---
    def calcular_hhi(self):
        hhi_index = (self.vw ** 2).sum(axis=1)
        activos_efectivos = 1 / hhi_index.replace(0, np.nan)

        if len(hhi_index) >= 12:
            hhi_interanual = (hhi_index / hhi_index.shift(12) - 1).dropna()
        else:
            hhi_interanual = pd.Series(dtype=float)
            

        self.hhi = hhi_index
        self.hhi_interanual = hhi_interanual
        self.activos_efectivos = activos_efectivos
















    # --- Cálculo de índice de sectores ---
    def calcular_indice_sectores(self):
        if self.sectores is None:
            raise ValueError("Error: debes descargar los sectores")

        precios_portafolio = self.precios_ajustados[self.tickers]
        retornos_portafolio = precios_portafolio.pct_change()
        mve = precios_portafolio.multiply(self.shares_outstanding, axis=1)
        mve_alineado = mve.reindex(retornos_portafolio.index, method='ffill')

        sectores_unicos = self.sectores.unique()
        fecha_inicio = retornos_portafolio.index[0]

        indice_sectores = pd.DataFrame(index=retornos_portafolio.index, columns=sectores_unicos)
        retornos_sectores_df = pd.DataFrame(index=retornos_portafolio.index, columns=sectores_unicos)

        for sector in sectores_unicos:
            tickers_del_sector = self.sectores[self.sectores == sector].index
            tickers_disponibles = [t for t in tickers_del_sector if t in retornos_portafolio.columns]

            if not tickers_disponibles:
                indice_sectores[sector] = np.nan
                retornos_sectores_df[sector] = np.nan
                continue

            mve_ticker_sector = mve_alineado[tickers_disponibles]
            total_mve_sector = mve_ticker_sector.sum(axis=1)
            pesos_sector = mve_ticker_sector.div(total_mve_sector, axis=0).fillna(0)
            retornos_sector = retornos_portafolio[tickers_disponibles]

            retornos_ponderados = (retornos_sector * pesos_sector.shift(1)).sum(axis=1)
            retornos_sectores_df[sector] = retornos_ponderados

            retornos_desde_inicio = retornos_ponderados.loc[fecha_inicio:]
            indice_sectores.loc[fecha_inicio:, sector] = (1 + retornos_desde_inicio).cumprod() * 100

        self.retornos_por_sector = retornos_sectores_df
        self.indice_sectores = indice_sectores
        return self.indice_sectores




    def calcular_frontera_eficiente(self):
        """Calcula los datos de la frontera eficiente por sectores"""
        # Usar retornos por sector
        retornos = self.retornos_por_sector.dropna(axis=1, how='all').dropna()
        
        # Matriz de covarianza
        self.V = retornos.cov().values
        
        # Vector iota (columna de unos)
        n = len(retornos.columns)
        self.iota = np.ones((n, 1))
        
        # Vector de medias
        self.mu = retornos.mean().values.reshape(-1, 1)
        
        # Vector de desviaciones estándar
        sigma_individual = retornos.std().values.reshape(-1, 1)

        # Matriz de covarianza inversa
        self.V1 = np.linalg.pinv(self.V)

        # Cálculos de la frontera eficiente
        self.A = self.iota.T @ self.V1 @ self.mu
        self.B = self.mu.T @ self.V1 @ self.mu
        self.C = self.iota.T @ self.V1 @ self.iota
        self.D = self.B * self.C - self.A**2
            
        # Cartera de mínima varianza (MVP)
        self.media_mvp = (self.A / self.C).item()
        self.volatilidad_mvp = np.sqrt((1 / self.C).item())

        # Frontera eficiente
        volatilidades = np.linspace(self.volatilidad_mvp, self.volatilidad_mvp * 2.5, 100)
        sqrt_term = np.sqrt(np.maximum(0, (self.D/self.C).item() * (volatilidades**2 - (1/self.C).item())))
        rendimientos_sup = (self.A/self.C).item() + sqrt_term  # Rama superior (+)
        rendimientos_inf = (self.A/self.C).item() - sqrt_term  # Rama inferior (-)

        return {
            'volatilidades': volatilidades,
            'rendimientos_sup': rendimientos_sup,
            'rendimientos_inf': rendimientos_inf,
            'media_mvp': self.media_mvp,
            'volatilidad_mvp': self.volatilidad_mvp,
            'mu_sectores': self.mu.flatten(),
            'sigma_sectores': sigma_individual,
            'sectores': retornos.columns.tolist()
        }
        



    









    ##CARTERAS g y h

    def carteras_gh(self):
        """
        Calcular las carteras fundamentales g y h de la teoría de Markowitz
        
        Siguiendo la lógica de spanning de carteras:
        Cualquier cartera eficiente puede expresarse como combinación lineal de g y h
        """
        # Verificar que las matrices estén calculadas
        if not hasattr(self, 'A') or self.A is None:
            raise ValueError("Primero ejecuta calcular_frontera_eficiente()")
        
        # Obtener matrices de la frontera eficiente
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        V1 = self.V1  # Matriz de covarianza inversa
        mu = self.mu  # Vector de rendimientos esperados
        iota = self.iota  # Vector de unos
        
        # Cálculo de las carteras g y h
        # CORRECCIÓN: A, B, C son escalares, no matrices
        g = (B.item() * V1 @ iota - A.item() * V1 @ mu) / D.item()
        h = (C.item() * V1 @ mu - A.item() * V1 @ iota) / D.item()
        g_h = g + h
        # Guardar como atributos de la clase
        self.cartera_g = g
        self.cartera_h = h

    
    def pesos_cartera_g_h(self):
        """
        Crea una tabla con los pesos de las carteras g, h y g+h por sectores
        
        Returns:
            pandas.DataFrame: Tabla con sectores y sus respectivos pesos
        """
        
        # Auto-calcular si no existen
        if not hasattr(self, 'cartera_g') or self.cartera_g is None:
            # Primero asegurar que existe la frontera eficiente
            if not hasattr(self, 'A') or self.A is None:
                self.calcular_frontera_eficiente()

            # Calcular las carteras
            self.carteras_gh()
        
        # Obtener los nombres de los sectores
        sectores = self.retornos_por_sector.dropna(axis=1, how='all').columns.tolist()
        
        # Convertir carteras a arrays 1D para facilitar manejo
        g_weights = self.cartera_g.flatten()
        h_weights = self.cartera_h.flatten()
        gh_weights = g_weights + h_weights
        
        # Crear DataFrame
        tabla_pesos = pd.DataFrame({
            'Sector': sectores,
            'Cartera_g': g_weights,
            'Cartera_h': h_weights,
            'Cartera_g+h': gh_weights
        })
        
        # Redondear para mejor visualización
        tabla_pesos['Cartera_g'] = tabla_pesos['Cartera_g']
        tabla_pesos['Cartera_h'] = tabla_pesos['Cartera_h']
        tabla_pesos['Cartera_g+h'] = tabla_pesos['Cartera_g+h']

        # Agregar fila de totales
        # Calcular totales como enteros
        total_g = int(round(tabla_pesos['Cartera_g'].sum(), 0))
        total_h = int(round(tabla_pesos['Cartera_h'].sum(), 0))
        total_gh = int(round(tabla_pesos['Cartera_g+h'].sum(), 0))
        
    # Agregar fila de totales
        totales = pd.DataFrame({
            'Sector': ['TOTAL'],
            'Cartera_g': [total_g],
            'Cartera_h': [total_h],
            'Cartera_g+h': [total_gh]
        })
        
        tabla_completa = pd.concat([tabla_pesos, totales], ignore_index=True)
        
        # Guardar como atributo para referencia futura
        self.tabla_pesos_gh = tabla_completa
        
        return tabla_completa





    






    ## CALCULAR CAPM DE SHARPE EMPIRICO Y TEORICO

    
    
    def camp_sharpe(self):
        # Obtener datos
        retornos_por_sector = self.retornos_por_sector
        sectores_exceso = self.sectores_exceso
        retornos_mercado = self.retornos_benchmark
        rf = self.rf
        portafolio = self.retornos_portafolio
        
        # FUNCIÓN PARA CONVERTIR ÍNDICES
        def convert_index_to_datetime(data):
            if hasattr(data, 'index'):
                if any(isinstance(idx, pd.Period) for idx in data.index):
                    data.index = data.index.to_timestamp()
                elif not isinstance(data.index, pd.DatetimeIndex):
                    try:
                        data.index = pd.to_datetime(data.index)
                    except:
                        pass
            return data
        
        # Convertir todos los índices
        sectores_exceso = convert_index_to_datetime(sectores_exceso.copy())
        retornos_mercado = convert_index_to_datetime(retornos_mercado.copy())
        rf = convert_index_to_datetime(rf.copy() if hasattr(rf, 'copy') else rf)
        portafolio = convert_index_to_datetime(portafolio.copy())
        
        # DEBUG: Verificar índices
      
        
        # Calcular excesos de retorno
        mve_exceso = portafolio.subtract(rf, axis=0)
        market_exceso = retornos_mercado.subtract(rf, axis=0)
        
        # Asignar nombres
        mve_exceso.name = 'Portafolio_Exceso'
        market_exceso.name = 'Benchmark_Exceso'
        
        # VARIANZAS
        var_rm = retornos_mercado.var(ddof=1)
        var_mve = portafolio.var(ddof=1)

        # DataFrames para resultados
        resultados_vs_benchmark = pd.DataFrame(
            columns=['Alpha', 'T_stat_Alpha', 'Beta_Empirico', 'T_stat_Beta', 'Beta_Teorico', 'R_squared'],
            index=sectores_exceso.columns
        )
        
        resultados_vs_portafolio = pd.DataFrame(
            columns=['Alpha', 'T_stat_Alpha', 'Beta_Empirico', 'T_stat_Beta', 'Beta_Teorico', 'R_squared'],
            index=sectores_exceso.columns
        )

        # ANÁLISIS 1: SECTORES vs BENCHMARK
        for sector in sectores_exceso.columns:
            try:
                # BETA TEÓRICO
                cov_sector_benchmark = retornos_mercado.cov(sectores_exceso[sector])
                beta_teorico = cov_sector_benchmark / var_rm
                
                # PREPARAR DATOS
                Y = sectores_exceso[sector]
                X = market_exceso
                
                # Asegurar que los índices coincidan
                aligned_data = pd.DataFrame({
                    'Y': Y,
                    'X': X
                }).dropna()
                
                if len(aligned_data) < 2:
                    print(f"Advertencia: Solo {len(aligned_data)} observaciones para {sector} vs Benchmark")
                    continue
                
                # REGRESIÓN
                X_sm = sm.add_constant(aligned_data['X'])
                model = sm.OLS(aligned_data['Y'], X_sm).fit()
                
                resultados_vs_benchmark.loc[sector] = [
                    model.params['const'],
                    model.tvalues['const'],
                    model.params['X'],
                    model.tvalues['X'],
                    beta_teorico,
                    model.rsquared
                ]
                
            except Exception as e:
                print(f"Error en {sector} vs Benchmark: {e}")
                continue

        # ANÁLISIS 2: SECTORES vs PORTAFOLIO
        for sector in sectores_exceso.columns:
            try:
                # BETA TEÓRICO
                cov_sector_portafolio = portafolio.cov(sectores_exceso[sector])
                beta_teorico = cov_sector_portafolio / var_mve
                
                # PREPARAR DATOS
                Y = sectores_exceso[sector]
                X = mve_exceso
                
                aligned_data = pd.DataFrame({
                    'Y': Y,
                    'X': X
                }).dropna()
                
                if len(aligned_data) < 2:
                    print(f"Advertencia: Solo {len(aligned_data)} observaciones para {sector} vs Portafolio")
                    continue
                
                # REGRESIÓN
                X_sm = sm.add_constant(aligned_data['X'])
                model = sm.OLS(aligned_data['Y'], X_sm).fit()
                
                resultados_vs_portafolio.loc[sector] = [
                    model.params['const'],
                    model.tvalues['const'],
                    model.params['X'],
                    model.tvalues['X'],
                    beta_teorico,
                    model.rsquared
                ]
                
            except Exception as e:
                print(f"Error en {sector} vs Portafolio: {e}")
                continue

        self.resultados_capm_vs_benchmark = resultados_vs_benchmark
        self.resultados_capm_vs_portafolio = resultados_vs_portafolio
        
        return resultados_vs_benchmark, resultados_vs_portafolio


   



















    # --- APARTADO DE GRAFICOS ---
    def graficar_indices(self):
        st.subheader("Índice Value Weighted vs Benchmark")
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(np.log(self.indice_portafolio), label="Portafolio", linewidth=2)
        ax.plot(np.log(self.indice_benchmark), label="Benchmark", linewidth=2)
        ax.set_title(f"Índice VW vs Índice {self.indice_benchmark.name}")
        ax.set_ylabel("Logaritmo del Índice")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    def graficar_hhi(self):
        st.subheader("Índice HHI")
        if len(self.tickers) > 1:
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(self.hhi, label="HHI")
            ax.set_title("Índice HHI")
            ax.legend()
            st.pyplot(fig, use_container_width=True)
        else: 
            st.info("Se necesitan al menos 2 activos para calcular y graficar el HHI.")

    def graficar_activos_efectivos(self):
        st.subheader("Activos Efectivos")
        if len(self.tickers) > 1:
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(self.activos_efectivos, label="Activos Efectivos")
            ax.set_title("Diversificación Efectiva")
            ax.legend()
            st.pyplot(fig, use_container_width=True)
        else:
            pass

    def graficar_hhi_interanual(self):
        if len(self.tickers) > 1:
            if not self.hhi_interanual.empty:
                st.subheader("Variación Interanual del HHI")
                fig, ax = plt.subplots(figsize=(20, 10))
                ax.plot(self.hhi_interanual * 100, label="HHI Variación Interanual")
                ax.axhline(y=0, color="black", linestyle="--")
                ax.legend()
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Datos insuficientes para calcular la variación interanual del HHI (se necesitan al menos 12 períodos).")
        else: 
            pass




    def graficar_indice_sectores(self):
        st.subheader("Índice Value Weighted por sector")
        fig, ax = plt.subplots(figsize=(20, 10))
        # Define la paleta de colores para los sectores
        num_sectores = len(self.indice_sectores.columns)
        colors = sns.color_palette("Paired", num_sectores)
        # Itera sobre los sectores y grafica cada uno con un color distinto de la paleta
        for i, sector in enumerate(self.indice_sectores.columns):
            ax.plot(self.indice_sectores[sector], label=sector, color=colors[i])

        ax.set_title("Índice por Sector")
        ax.legend()
        st.pyplot(fig, use_container_width=True)




    ##GRAFICO APILADOS Y EVOLUCION DE PESOS DE CADA SECTOR EN EL PORTAFOLIO
    def graficos_apilados(self):
        st.subheader("Evolución de Pesos por Sector")
        if self.vw is None or self.sectores is None:
            st.warning("Error: No se puede generar el gr")

        ##PESOS DE CADA SECTOR
        pesos_por_sector = pd.DataFrame(index=self.vw.index)
        sectores_unicos = self.sectores.unique()
        for sector in sectores_unicos:
            tickers_del_sector =  self.sectores[self.sectores == sector].index
            pesos_por_sector[sector] = self.vw[tickers_del_sector].sum(axis=1)

        # Reordenar las columnas del grafico
        sectores_ordenados = pesos_por_sector.iloc[-1].sort_values(ascending=False).index
        peso_por_sector = pesos_por_sector[sectores_ordenados]
        ## Stackplot
        fig, ax = plt.subplots(figsize=(20, 10))
        colors = sns.color_palette("Paired", len(sectores_ordenados))
        ax.stackplot(
            pesos_por_sector.index,
            pesos_por_sector.values.T,
            labels=pesos_por_sector.columns,
            colors=colors
        )
        ax.set_title('Composición del Portafolio por Sector a lo largo del Tiempo', fontsize=20)
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Peso en el Portafolio (%)')
        ax.legend(title='Sectores', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)









    "GRAFICOS DE QUANSTATS"

        ## Cálculo del Sharpe Ratio Móvil Portafolio
    def rolling_sharpe_portafolio(self):
        retornos = self.retornos_portafolio
        
        # Verificar que existen datos
        if retornos is None or len(retornos) == 0:
            st.info("No hay datos de portafolio disponibles")
            return
        
        # Verificar datos suficientes para rolling (mínimo 30, recomendado 126)
        min_datos = 126

        
        if len(retornos) < min_datos:
            st.info(f"Datos insuficientes para Rolling Sharpe del portafolio.")
            st.info(f"Se necesitan al menos {min_datos} observaciones. Disponibles: {len(retornos)}")
            
            # Mostrar Sharpe simple en su lugar
            try:
                sharpe_simple = qs.stats.sharpe(retornos)
                st.metric("Sharpe Ratio Simple - Portafolio", f"{sharpe_simple:.3f}")
            except:
                st.error("No se pudo calcular Sharpe ratio")
            return
    
        
        st.subheader("Rolling Sharpe Ratio Portafolio")
        
        
        try:
            fig = qs.plots.rolling_sharpe(retornos, show=False)
            st.pyplot(fig)
            plt.close(fig)
            sharpe_simple = qs.stats.sharpe(retornos)
            st.metric("Sharpe Ratio simple:", f"{sharpe_simple:.3f}")
        except Exception as e:
            st.error(f"Error generando Rolling Sharpe")
            
            # Fallback a Sharpe simple
            try:
                sharpe_simple = qs.stats.sharpe(retornos)
                st.metric("Sharpe Ratio simple:", f"{sharpe_simple:.3f}")
            except:
                pass















    ##Rolling sharpe del benchhmark

    def rolling_sharpe_benchmark(self):
        benchmark = self.retornos_benchmark
        
        # Verificar que existen datos
        if benchmark is None or len(benchmark) == 0:
            st.warning("No hay datos de benchmark disponibles")
            return
        
        # Verificar datos suficientes para rolling
        min_datos = 126
        
        if len(benchmark) < min_datos:
            st.info(f"Datos insuficientes para Rolling Sharpe del benchmark.")
            st.info(f"Se necesitan al menos {min_datos} observaciones. Disponibles: {len(benchmark)}")
            
            # Mostrar Sharpe simple en su lugar
            try:
                sharpe_simple = qs.stats.sharpe(benchmark)
                st.metric(f"Sharpe Ratio Simple - {self.benchmark}", f"{sharpe_simple:.3f}")
            except:
                st.error("No se pudo calcular Sharpe ratio")
            return
        


        st.subheader(f"Rolling Sharpe Ratio {self.indice_benchmark.name}")
       
        try:
            fig = qs.plots.rolling_sharpe(benchmark, show=False)
            st.pyplot(fig)
            plt.close(fig)
            sharpe_simple = qs.stats.sharpe(benchmark)
            st.metric("Sharpe Ratio simple:", f"{sharpe_simple:.3f}")
        except Exception as e:
            st.error(f"Error generando Rolling Sharpe")
            
            # Fallback a Sharpe simple
            try:
                sharpe_simple = qs.stats.sharpe(benchmark)
                st.metric("Sharpe Ratio simple:", f"{sharpe_simple:.3f}")
            except:
                pass

















 ## Cálculo del Drawdown Portafolio
    def rolling_drawdown_portafolio(self):
        retornos = self.retornos_portafolio
        
        # Verificar que existen datos
        if retornos is None or len(retornos) == 0:
            st.warning("No hay datos de portafolio disponibles")
            return
        
       
    
        st.subheader("Drawdown Portafolio")
        try:
            fig = qs.plots.drawdown(retornos, show=False)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error generando Underwater Plot")








    ##Rolling sharpe del benchhmark

    def rolling_drawdown_benchmark(self):
        benchmark = self.retornos_benchmark
        
        # Verificar que existen datos
        if benchmark is None or len(benchmark) == 0:
            st.warning("No hay datos de benchmark disponibles")
            return
        
        st.subheader(f"Drawdown {self.indice_benchmark.name}")
        try:
            fig = qs.plots.drawdown(benchmark, show=False)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error generando Underwater Plot")






    "Graficos frontera eficiente"
    
    def graficar_frontera_eficiente(self):
        """Grafica la frontera eficiente"""
        datos = self.calcular_frontera_eficiente()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Frontera eficiente
        ax.plot(datos['volatilidades'], datos['rendimientos_sup'], 'b-', label='Frontera Eficiente')
        ax.plot(datos['volatilidades'], datos['rendimientos_inf'], 'b--', label='Frontera Ineficiente')
        
        # MVP
        ax.scatter(datos['volatilidad_mvp'], datos['media_mvp'], color='red', s=100, label='MVP', zorder=5)
        
        # Sectores individuales
        ax.scatter(datos['sigma_sectores'], datos['mu_sectores'], color='green', s=80, label='Sectores', alpha=0.7)
        
        # Etiquetas de sectores
        for i, sector in enumerate(datos['sectores']):
            ax.annotate(sector, (datos['sigma_sectores'][i], datos['mu_sectores'][i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Volatilidad')
        ax.set_ylabel('Rendimiento Esperado')
        ax.set_title('Frontera Eficiente por Sectores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close(fig)




   
     































    # --- ANALISIS ---
    def ejecutar_analisis_completo(self):
        self.descargar_datos()
        self.calcular_retornos()
        self.calcular_hhi()
        self.calcular_indice_sectores()     


















# --- INTERFAZ STREAMLIT ---
st.title("Análisis de Portafolio")

# Selección de tickers
input_method = st.radio(
    "Selecciona el método de entrada para los Tickers:",
    ("Entrada de texto", "Subir archivo de texto (.txt)")
)

tickers = []
MAX_TICKERS = 60  # Límite máximo de tickers

if input_method == "Entrada de texto":
    tickers_input = st.text_area("Tickers del Portafolio (separados por coma)", "AAPL, MSFT, GOOGL",)
    tickers = [t.strip().upper() for t in tickers_input.replace(',', '\n').split('\n') if t.strip()]
    
    # Verificar límite para entrada de texto
    if len(tickers) > MAX_TICKERS:
        st.warning(f"Has ingresado {len(tickers)} tickers. El máximo permitido es {MAX_TICKERS}.")
        st.info(f"Se procesarán solo los primeros {MAX_TICKERS} tickers.")
        tickers = tickers[:MAX_TICKERS]  # Tomar solo los primeros 60
        
else:
    uploaded_file = st.file_uploader("Sube un archivo de texto con los tickers (separados por coma o por línea)", type="txt")
    if uploaded_file is not None:
        try:
            string_data = uploaded_file.read().decode("utf-8")
            processed_data = string_data.replace('\n', ',')
            tickers = [t.strip().upper() for t in processed_data.split(',') if t.strip()]
            
            if not tickers:
                st.warning("El archivo está vacío o no contiene tickers válidos.")
            elif len(tickers) > MAX_TICKERS:
                st.warning(f"El archivo contiene {len(tickers)} tickers. El máximo permitido es {MAX_TICKERS}.")
                st.info(f"Se procesarán solo los primeros {MAX_TICKERS} tickers.")
                tickers = tickers[:MAX_TICKERS]  # Tomar solo los primeros 60
                
        except Exception as e:
            st.error(f"Ocurrió un error al leer el archivo: {e}")

# Benchmark
benchmark_input = st.text_input("Benchmark (Solo uno)", "^GSPC").upper()
benchmarks = [b.strip() for b in benchmark_input.replace(',', ' ').split() if b.strip()]
st.info('Asegúrate de que el benchmark y los tickers sean válidos en Yahoo Finance.')
if len(benchmarks) > 1:
    st.warning("Error: Por favor ingresa solo un benchmark.")
    # Puedes detener la ejecución del script si el usuario ha cometido un error
    st.stop()
else:
    benchmark = benchmarks[0]

# Fechas
st.markdown("---")
st.subheader("Selección de Período")
col1, col2 = st.columns(2)

with col1:
    # Año de inicio
    years_start = list(range(1920, datetime.today().year + 1))
    default_index_start = min(120, len(years_start) - 1)
    start_year = st.selectbox("Año de inicio", options=years_start, index=default_index_start)

    # Mes de inicio
    months = list(range(1, 13))
    start_month = st.selectbox("Mes de inicio", options=months, format_func=lambda x: calendar.month_name[x], index=0)

    # Fecha de inicio
    start_date = datetime(start_year, start_month, 1)

with col2:
    # Año final
    years_end = list(range(1920, datetime.today().year + 1))
    default_index_end = len(years_end) - 1
    end_year = st.selectbox("Año final", options=years_end, index=default_index_end)

    # Mes final
    end_month = st.selectbox(
        "Mes final", 
        options=months, 
        format_func=lambda x: calendar.month_name[x], 
        index=datetime.today().month - 1
    )

    # Fecha final (último día del mes seleccionado)
    end_date = datetime(end_year, end_month, 1) - timedelta(days=1)
if start_date >= end_date:
    st.info("La fecha de inicio debe ser anterior a la fecha final.")
    st.stop()









# Botones de análisis y limpieza
col1, col2 = st.columns(2)

with col1:
    if st.button("Realizar Análisis", type="secondary"):
        if tickers and benchmark:
            with st.spinner("Descargando y procesando datos..."):
                analyzer = PortafolioAnalyzer(tickers, benchmark, start_date, end_date)
                analyzer.ejecutar_analisis_completo()
                st.session_state['analyzer'] = analyzer
            
            st.success("Análisis completado.")
        else:
            st.error("Por favor ingresa o sube al menos un ticker y un benchmark válido.")

with col2:
    if st.button(" Limpiar Datos", type="secondary"):
        st.session_state.clear()
        st.rerun()




if 'analyzer' in st.session_state:
    analyzer = st.session_state['analyzer']

    # ORGANIZACIÓN DE GRÁFICOS CON EL SELECTBOX
    selected_tab = st.selectbox(
        "Selecciona una sección de análisis:",
        ["Índice VW vs Benchmark", "Herfindahl-Hirschman", "Sectores", "Quantstats","Frontera Eficiente","FFC - CAPM"]
    )

    if selected_tab == "Índice VW vs Benchmark":
        st.header("Comparación de Índices")
        analyzer.graficar_indices()

    elif selected_tab == "Herfindahl-Hirschman":
        st.header("Análisis de Concentración")
        col_hhi1, col_hhi2 = st.columns(2)
        with col_hhi1:
            analyzer.graficar_hhi()
        with col_hhi2:
            analyzer.graficar_activos_efectivos()
        analyzer.graficar_hhi_interanual()
        st.subheader("Métricas")
        if analyzer.hhi is not None:
            col_met1, col_met2, col_met3 = st.columns(3)
            with col_met1:
                st.metric("Último HHI", f"{analyzer.hhi.iloc[-1]:.4f}")
            with col_met2:
                st.metric("Activos Efectivos", f"{analyzer.activos_efectivos.iloc[-1]:.2f}")
            with col_met3:
                if analyzer.hhi_interanual is not None and not analyzer.hhi_interanual.empty:
                    st.metric("Última Variación Interanual", f"{analyzer.hhi_interanual.iloc[-1]*100:.2f}%")

    elif selected_tab == "Sectores":
        st.header("Información de Sectores")
        col_sec1, col_sec2 = st.columns(2)
        with col_sec1:
            analyzer.graficar_indice_sectores()
        with col_sec2:
            analyzer.graficos_apilados()
            


    elif selected_tab == "Frontera Eficiente":
        if analyzer.sectores.nunique() < 2:
            st.info("Se necesitan al menos 2 sectores para los cálculos de la frontera eficiente.")
        else:
            st.header("Análisis de Frontera Eficiente")
            analyzer.graficar_frontera_eficiente()
            mvef1, mvef2 = st.columns(2)
            with mvef1:
                st.metric("Rendimiento Portafolio Mínima Varianza", f"{analyzer.media_mvp:.2f}%")
            with mvef2:
                st.metric("Riesgo (Volatilidad) Portafolio Mínima Varianza", f"{analyzer.volatilidad_mvp:.2f}%")
            st.header(" Carteras g y h")
            tabla = analyzer.pesos_cartera_g_h()
            st.dataframe(tabla, use_container_width=True)




    elif selected_tab == "FFC - CAPM":
        st.header("Factores Fama-French")
        analyzer.descargar_factores_fama_french()  # Esto llena self.fama
        analyzer.unir_factores_y_retornos()
        analyzer.resumen_coeficientes()
        analyzer.regresiones_ff()
        st.dataframe(analyzer.df_resumen, use_container_width=True)

        st.header("Resultados del CAPM de Sharpe")
        analyzer.camp_sharpe()
        st.subheader("Sectores vs Benchmark")
        st.dataframe(analyzer.resultados_capm_vs_benchmark, use_container_width=True)

        st.subheader("Sectores vs Portafolio")
        st.dataframe(analyzer.resultados_capm_vs_portafolio, use_container_width=True)









    elif selected_tab == 'Quantstats':
        st.header("Análisis Quantstats")
        quants_sec1, quants_sec2 = st.columns(2)
        with quants_sec1:
            analyzer.rolling_sharpe_portafolio()
            analyzer.rolling_drawdown_portafolio()
        with quants_sec2:
            analyzer.rolling_sharpe_benchmark()
            analyzer.rolling_drawdown_benchmark()

   
st.markdown("---")
st.info(" **Nota:** El botón 'Limpiar Datos' borra todos los gráficos y resultados anteriores, permitiéndote comenzar un nuevo análisis.")