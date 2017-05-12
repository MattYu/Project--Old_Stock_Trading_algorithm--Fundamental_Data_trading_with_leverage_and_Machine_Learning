"""
Strategies and inspiration: Sentdex youtube
@Ming T. Yu
@2016
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from collections import Counter
import numpy as np

 
def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    
    context.limit = 10
    context.historical_bars = 1000
    context.feature_window = 20
    schedule_function(rebalance, 
                      date_rule = date_rules.every_day(),
                      time_rule=time_rules.market_open(hours=1))    
    set_commission(commission.PerShare(cost=0.005))  
    
def before_trading_start(context):
    context.fundamentals = get_fundamentals(
        query(
            fundamentals.valuation_ratios.pb_ratio,
            fundamentals.valuation_ratios.pe_ratio,
        )
        .filter(
            fundamentals.valuation_ratios.pe_ratio < 14
        )
        .filter(
            fundamentals.valuation_ratios.pb_ratio < 2 
        )
        .order_by(
            fundamentals.valuation.market_cap.desc()
        )
        .limit(context.limit)
    )
    
    print (('Potentials buys:', context.fundamentals)) 
    context.assets = [context.fundamentals.columns.values]

 
def rebalance(context,data):
    """
    Called every minute.
    """
    cash = context.portfolio.cash
    current_position = context.portfolio.positions
    
    for stock in context.fundamentals:
        prices = data.history(stock, 'price', context.historical_bars,'1d')
        price_hist1= data.history(stock, 'price', 60, '1d')
        price_hist2= data.history(stock, 'price', 200, '1d')
        Ma1= price_hist1.mean()
        Ma2= price_hist2.mean()
        price = data.current(stock, 'price')
        
        start_bar = context.feature_window
        price_list = prices.tolist()
                
        X = []
        y = []
        
        bar = start_bar
        
        while bar < len(price_list)-1:
            try:
                end_price = price_list[bar+1]
                begin_price = price_list[bar]
                
                print bar
                
                pricing_list = []
                xx = 0
                for _ in range(context.feature_window):
                    price = price_list[bar-(context.feature_window-xx)]
                    pricing_list.append(price)
                    xx += 1
                    
                features = np.around(np.diff(pricing_list)/ pricing_list[:-1]* 100.0, 1)
                               
                if end_price> begin_price:
                    label = 1
                else:
                    label = -1
                
                bar +=1
                X.append(features)
                y.append(label)                       
                
            except Exception as e:
                bar +=1
                print(str(e))
         
        clf1 = RandomForestClassifier()
        clf2 = LinearSVC()
        clf3 = NuSVC()
        clf4 = LogisticRegression()
        
        last_prices = price_list[-context.feature_window:]
        current_features = np.around(np.diff(pricing_list)/ last_prices[:-1]* 100.0, 1)
        
        X.append(current_features)
        X= preprocessing.scale(X)
        
        current_features = X[-1]
        X = X[:-1]
        
        clf1.fit(X,y)
        clf2.fit(X,y)
        clf3.fit(X,y)
        clf4.fit(X,y)



        
        p1= clf1.predict(current_features)[0]
        p2= clf2.predict(current_features)[0]
        p3= clf3.predict(current_features)[0]
        p4= clf4.predict(current_features)[0]
        
        
        #if Counter([p1, p2, p3, p4]).most_common(1)[0][1] >= 4:
            #p = Counter([p1, p2, p3, p4]).most_common(1)[0][0]
            
        #else:
            #p = 0
        #print(('prediction',p))
        
        Su = sum([p1, p2, p3, p4])
        if Su == 4:
            p = 1
        elif Su <= -2:
            p = -1
        else:
            p = 0
        current_position = context.portfolio.positions[stock].amount
        stock_price = data.current(stock, 'price')
        plausible_investment= cash/10
        stop_price = stock_price - (stock_price * 0.01)
        
        share_amount= int(plausible_investment/stock_price)
        
        try:
            if current_position == 0:
                if context.fundamentals[stock]['pe_ratio']< 11 or (Ma1>Ma2) or p ==1:
                    order(stock, share_amount, style=StopOrder(stop_price))
            if current_position > 0:
                if stock not in context.fundamentals or ((Ma1<Ma2) and p == -1):
                    order_target_percent(stock, 0)
        except Exception as e:
                print(str(e))
                
        record('Ma1', Ma1)
        record('Ma2', Ma2)
        record('leverage', context.account.leverage)