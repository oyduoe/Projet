import yfinance as yf 
import numpy as np
import pandas as pd

full_dict = {
    "EQT": "EQT Corporation",
    "SAGA-B.ST": "AB Sagax",
    "ACGBY": "Agricultural Bank of China Limited",
    "ATEYY": "Advantest Corporation",
    "2395.TW": "Advantech Co., Ltd.",
    "ADM.L": "Admiral Group plc",
    "AFL": "Aflac Incorporated",
    "ANET": "Arista Networks, Inc.",
    "ARES": "Ares Management Corporation",
    "ACGL": "Arch Capital Group Ltd.",
    "300999.SZ": "Yihai Kerry Arawana Holdings Co., Ltd.",
    "AU": "AngloGold Ashanti plc",
    "AIR.PA": "Airbus SE",
    "2618.TW": "EVA Airways Corporation",
    "AON": "Aon plc",
    "A17U.SI": "CapitaLand Ascendas REIT",
    "TEMN.SW": "Temenos AG",
    "HOLN.SW": "Holcim Ltd",
    "2802.T": "Ajinomoto Co., Inc.",
    "MB.MI": "Mediobanca Banca di Credito Finanziario S.p.A.",
    "BAMI.MI": "Banco BPM S.p.A.",
    "PST.MI": "Poste Italiane S.p.A.",
    "ALE.WA": "Allegro.eu SA",
    "AVB": "AvalonBay Communities, Inc.",
    "AVY": "Avery Dennison Corporation",
    "2588.HK": "BOC Aviation Limited",
    "INDIGO.NS": "InterGlobe Aviation Ltd",
    "9202.T": "ANA Holdings Inc.",
    "KMI": "Kinder Morgan, Inc."
}

currency_dict = {
    "EQT": "USD",
    "SAGA-B.ST": "SEK",
    "ACGBY": "USD",
    "ATEYY": "USD",
    "2395.TW": "TWD",
    "ADM.L": "GBP",
    "AFL": "USD",
    "ANET": "USD",
    "ARES": "USD",
    "ACGL": "USD",
    "300999.SZ": "CNY",
    "AU": "USD",
    "AIR.PA": "EUR",
    "2618.TW": "TWD",
    "AON": "USD",
    "A17U.SI": "SGD",
    "TEMN.SW": "CHF",
    "HOLN.SW": "CHF",
    "2802.T": "JPY",
    "MB.MI": "EUR",
    "BAMI.MI": "EUR",
    "PST.MI": "EUR",
    "ALE.WA": "PLN",
    "AVB": "USD",
    "AVY": "USD",
    "2588.HK": "HKD",
    "INDIGO.NS": "INR",
    "9202.T": "JPY",
    "KMI": "USD"
}

def devisechange(df :pd.DataFrame,df_forex: pd.DataFrame, target_curr: str) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index).date
    df_forex.index = pd.to_datetime(df_forex.index).date
    forex_filtré = df_forex.loc[df_forex.index.isin(df.index)]
    df_converted = df.copy()
    stock_curr=0 # pour stocker la devise actuelle du stock
    for stock in df_converted.columns:
        stock_curr = currency_dict[stock]
        if stock_curr != target_curr:
            found= False
            for change in forex_filtré.columns:
                if change == f"{stock_curr}{target_curr}":
                    df_converted[stock] = df_converted[stock] * forex_filtré[change]
                    found=True
                    break
                elif change == f"{target_curr}{stock_curr}":
                    df_converted[stock] = df_converted[stock]/forex_filtré[change]
                    found=True
                    break
            if not found:
                print(f"Taux de change non trouvé pour {stock_curr} -> {target_curr}")
    return df_converted

tickers=list(full_dict.keys())
price_data = yf.download(tickers, start='2023-01-01', end='2024-12-31')['Close'].ffill()

df_forex = pd.read_csv("Devise.csv", index_col=0)
df_forex.index = pd.to_datetime(df_forex.index)

price_data_converted = devisechange(price_data,df_forex, "EUR")

returns_data = price_data_converted.pct_change().replace([np.inf, -np.inf], 0)
returns_data.index = pd.to_datetime(returns_data.index)
returns_data = returns_data[returns_data.index.to_series().dt.dayofweek < 5].dropna(axis=0)
returns_data.to_csv("Data.csv")