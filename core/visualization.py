import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from core.sector_analysis import SectorType

def display_summary_dashboard(df):
    """Display enhanced summary dashboard"""
    st.subheader("📊 Market Overview")
    
    # Enhanced metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_score = df['Valuation Score'].mean()
        delta = avg_score - 50
        st.metric(
            "Average Valuation Score", 
            f"{avg_score:.1f}",
            delta=f"{delta:.1f} vs Neutral",
            help="Scores above 50 indicate undervaluation"
        )
    
    with col2:
        best_stock = df.loc[df['Valuation Score'].idxmax()]
        st.metric(
            "Top Pick", 
            best_stock['Ticker'],
            f"Score: {best_stock['Valuation Score']:.1f}",
            help="Stock with the highest overall score"
        )
    
    with col3:
        avg_tech_score = df['Technical Score'].mean()
        tech_delta = avg_tech_score - 50
        st.metric(
            "Average Technical Score",
            f"{avg_tech_score:.1f}",
            delta=f"{tech_delta:.1f} vs Neutral",
            help="Technical strength indicator"
        )
    
    with col4:
        avg_volatility = df['30D Volatility'].mean() if '30D Volatility' in df.columns else 0
        st.metric(
            "Average 30D Volatility",
            f"{avg_volatility:.2%}",
            help="Average annualized volatility across selected stocks"
        )
    
    # Market interpretation
    if avg_score > 60:
        st.success("🔥 Overall market segment appears undervalued")
    elif avg_score < 40:
        st.warning("⚠️ Overall market segment appears overvalued")
    else:
        st.info("📊 Overall market segment appears fairly valued")
    
    # Enhanced recommendation distribution
    st.subheader("Investment Recommendations")
    rec_df = df.groupby('Recommendation').size().reset_index(name='Count')
    fig_rec = px.pie(
        rec_df, 
        values='Count', 
        names='Recommendation',
        color='Recommendation',
        color_discrete_map={
            'Strong Buy': '#2E7D32',
            'Buy': '#4CAF50',
            'Hold': '#FFC107',
            'Sell': '#F44336',
            'Strong Sell': '#B71C1C'
        },
        hole=0.4
    )
    st.plotly_chart(fig_rec)
    
    # Key insights
    display_key_insights(df)

def display_technical_analysis(df):
    """Display enhanced technical analysis"""
    st.subheader("Technical Indicators")
    
    # Add technical analysis explanation
    with st.expander("Understanding Technical Indicators"):
        st.markdown("""
        - **RSI (Relative Strength Index)**: Momentum indicator showing overbought/oversold conditions
        - **Moving Averages**: Trend indicators showing short-term vs long-term price movements
        - **Volume**: Trading activity indicator
        """)
    
    # RSI Analysis with interpretation
    st.subheader("RSI Analysis")
    if 'RSI' in df.columns:
        fig_rsi = px.scatter(
            df,
            x='Ticker',
            y='RSI',
            color='Recommendation',
            title='Relative Strength Index (RSI)',
            color_discrete_map={
                'Strong Buy': '#2E7D32',
                'Buy': '#4CAF50',
                'Hold': '#FFC107',
                'Sell': '#F44336',
                'Strong Sell': '#B71C1C'
            }
        )
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        st.plotly_chart(fig_rsi)
    else:
        st.warning("RSI data not available in the dataset")
    
    # Moving Averages with crossover analysis
    st.subheader("Moving Average Analysis")
    if 'MA50' in df.columns and 'MA200' in df.columns:
        ma_data = df[['Ticker', 'MA50', 'MA200']].melt(
            id_vars=['Ticker'],
            var_name='MA Type',
            value_name='Value'
        )
        fig_ma = px.line(
            ma_data,
            x='Ticker',
            y='Value',
            color='MA Type',
            title='Moving Averages Comparison'
        )
        st.plotly_chart(fig_ma)
        
        # Add MA crossover signals
        for _, row in df.iterrows():
            if row['MA50'] > row['MA200']:
                st.info(f"🔵 {row['Ticker']}: Golden Cross (Bullish Signal)")
            elif row['MA50'] < row['MA200']:
                st.warning(f"🔴 {row['Ticker']}: Death Cross (Bearish Signal)")
    else:
        st.warning("Moving Average data not available in the dataset")

def display_stock_comparison(df):
    """Display stock comparison analysis"""
    st.subheader("📊 Stock Comparison")
    
    # Select stocks to compare
    stocks_to_compare = st.multiselect(
        "Select stocks to compare (max 3)",
        df['Ticker'].tolist(),
        max_selections=3
    )
    
    if stocks_to_compare:
        comparison_df = df[df['Ticker'].isin(stocks_to_compare)]
        
        # Radar chart comparison
        metrics = ['Valuation Score', 'Technical Score', 'Sector Score']
        # Check if all metrics are available
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if len(available_metrics) > 0:
            fig = go.Figure()
            
            for ticker in stocks_to_compare:
                stock_data = comparison_df[comparison_df['Ticker'] == ticker]
                fig.add_trace(go.Scatterpolar(
                    r=[stock_data[metric].iloc[0] if metric in stock_data.columns else 0 for metric in available_metrics],
                    theta=available_metrics,
                    fill='toself',
                    name=ticker
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Comparative Analysis"
            )
            st.plotly_chart(fig)
        else:
            st.warning("Required metrics not available for comparison")
        
        # Detailed comparison table
        st.subheader("Detailed Comparison")
        comparison_metrics = ['Ticker', 'Current Price', 'Valuation Score', 
                            'Technical Score', 'Sector Score', 'Recommendation']
        # Filter to available columns
        available_comparison_metrics = [m for m in comparison_metrics if m in comparison_df.columns]
        st.dataframe(comparison_df[available_comparison_metrics])

def display_key_insights(df):
    """Display key insights from analysis"""
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Best Value Opportunities")
        if 'Valuation Score' in df.columns:
            best_value = df.nlargest(3, 'Valuation Score')
            for _, stock in best_value.iterrows():
                st.write(f"**{stock['Ticker']}**: Score {stock['Valuation Score']:.1f}")
                if 'Recommendation' in stock:
                    st.write(f"Recommendation: {stock['Recommendation']}")
                st.write("---")
        else:
            st.warning("Valuation Score not available")
    
    with col2:
        st.markdown("#### Technical Standouts")
        if 'Technical Score' in df.columns:
            best_tech = df.nlargest(3, 'Technical Score')
            for _, stock in best_tech.iterrows():
                st.write(f"**{stock['Ticker']}**: Score {stock['Technical Score']:.1f}")
                if 'RSI' in stock:
                    st.write(f"RSI: {stock.get('RSI', 'N/A')}")
                st.write("---")
        else:
            st.warning("Technical Score not available")

def display_detailed_analysis(df):
    """Display detailed analysis with filtering"""
    st.subheader("Comprehensive Analysis")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        if 'Valuation Score' in df.columns:
            min_score = st.slider("Minimum Valuation Score", 0, 100, 0)
        else:
            min_score = 0
            st.warning("Valuation Score not available for filtering")
    
    with col2:
        if 'Recommendation' in df.columns:
            selected_recommendations = st.multiselect(
                "Filter by Recommendation",
                df['Recommendation'].unique().tolist(),
                default=df['Recommendation'].unique().tolist()
            )
        else:
            selected_recommendations = []
            st.warning("Recommendation data not available for filtering")
    
    # Filter DataFrame
    if 'Valuation Score' in df.columns and len(selected_recommendations) > 0:
        filtered_df = df[
            (df['Valuation Score'] >= min_score) &
            (df['Recommendation'].isin(selected_recommendations))
        ]
    elif 'Valuation Score' in df.columns:
        filtered_df = df[df['Valuation Score'] >= min_score]
    elif len(selected_recommendations) > 0:
        filtered_df = df[df['Recommendation'].isin(selected_recommendations)]
    else:
        filtered_df = df
    
    # Display filtered results
    display_cols = ['Ticker', 'Company Name', 'Sector', 'Current Price', 
                   'Valuation Score', 'Technical Score', 'Sector Score',
                   '30D Volatility', 'Recommendation']
    
    # Filter for available columns
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    
    if len(available_cols) > 0:
        # Apply styling conditionally based on available columns
        numeric_cols = [col for col in ['Valuation Score', 'Technical Score', 'Sector Score'] 
                       if col in filtered_df.columns]
        
        styled_df = filtered_df[available_cols].style
        
        if numeric_cols:
            styled_df = styled_df.background_gradient(subset=numeric_cols, cmap='RdYlGn')
            
        if '30D Volatility' in filtered_df.columns:
            styled_df = styled_df.background_gradient(subset=['30D Volatility'], cmap='YlOrRd')
            
        if 'Recommendation' in filtered_df.columns:
            styled_df = styled_df.applymap(color_recommendation, subset=['Recommendation'])
        
        st.dataframe(styled_df)
    else:
        st.warning("No columns available to display")
    
    # Export functionality
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Analysis Results",
        data=csv,
        file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def display_sector_analysis(df, sector_metrics):
    """Display enhanced sector analysis"""
    st.subheader("Sector Analysis")
    
    # Get unique sectors
    if 'Sector' in df.columns:
        sectors = df['Sector'].unique().tolist()
        
        # Sector selector
        selected_sector = st.selectbox("Select Sector for Analysis", sectors)
        
        # Filter for selected sector
        sector_df = df[df['Sector'] == selected_sector]
        
        if not sector_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Sector metrics
                st.markdown(f"### {selected_sector} Metrics")
                if 'Sector Score' in sector_df.columns:
                    avg_sector_score = sector_df['Sector Score'].mean()
                    st.metric(
                        "Average Sector Score",
                        f"{avg_sector_score:.1f}",
                        delta=f"{avg_sector_score - 50:.1f} vs Market",
                        help="Sector-specific performance score"
                    )
                
                # Get sector-specific metrics
                try:
                    sector_type = map_sector_to_type(selected_sector)
                    if sector_type in sector_metrics.SECTOR_CONFIGS:
                        specific_metrics = sector_metrics.SECTOR_CONFIGS[sector_type]['specific_metrics']
                        st.write("Key Sector Metrics:")
                        
                        # Display available metrics for the sector
                        metrics_found = False
                        for metric, config in specific_metrics.items():
                            # Convert metric name to the format used in your DataFrame
                            df_metric_name = metric.lower()
                            if df_metric_name in sector_df.columns:
                                metrics_found = True
                                avg_value = sector_df[df_metric_name].mean()
                                threshold = config['threshold']
                                if threshold is not None:
                                    st.metric(
                                        metric.replace('_', ' ').title(),
                                        f"{avg_value:.2f}",
                                        delta=f"{avg_value - threshold:.2f} vs Threshold"
                                    )
                        
                        if not metrics_found:
                            st.info(f"No specific metrics available for {selected_sector}")
                except Exception as e:
                    st.error(f"Error displaying sector metrics: {e}")
            
            with col2:
                # Sector risk analysis
                st.markdown("### Risk Analysis")
                try:
                    sector_type = map_sector_to_type(selected_sector)
                    if sector_type in sector_metrics.SECTOR_CONFIGS:
                        risk_factors = sector_metrics.SECTOR_CONFIGS[sector_type]['risk_factors']
                        st.write("Key Risk Factors:")
                        
                        # Display available risk factors
                        risks_found = False
                        for risk in risk_factors:
                            risk_col = f"{risk.lower()}_risk"
                            if risk_col in sector_df.columns:
                                risks_found = True
                                risk_value = sector_df[risk_col].mean()
                                st.progress(risk_value/100)
                                st.write(f"{risk.replace('_', ' ')}: {risk_value:.1f}%")
                        
                        if not risks_found:
                            st.info(f"No risk factors available for {selected_sector}")
                except Exception as e:
                    st.error(f"Error displaying risk analysis: {e}")
            
            # Sector performance visualization
            st.subheader("Sector Performance Distribution")
            performance_metrics = [m for m in ['Valuation Score', 'Technical Score', 'Sector Score'] 
                                  if m in sector_df.columns]
            
            if performance_metrics:
                fig = go.Figure()
                for metric in performance_metrics:
                    fig.add_trace(go.Box(
                        y=sector_df[metric],
                        name=metric,
                        boxpoints='all'
                    ))
                fig.update_layout(
                    title=f"{selected_sector} Score Distribution",
                    yaxis_title="Score",
                    showlegend=True
                )
                st.plotly_chart(fig)
            else:
                st.warning("No performance metrics available for visualization")
        else:
            st.warning("No data available for selected sector")
    else:
        st.error("Sector data not available in the dataset")

def create_sector_visualizations(metrics, sector_type, sector_metrics):
    """Create comprehensive sector-specific visualizations"""
    figures = {}
    
    # Get sector metrics breakdown
    breakdown = {
        'base_metrics': {},
        'sector_specific': {},
        'risk_factors': {}
    }
    
    # 1. Radar Chart for Base Metrics
    base_metrics_fig = go.Figure()
    
    metrics_values = []
    metrics_thresholds = []
    metric_names = []
    
    # Check if we have base metrics
    if len(breakdown['base_metrics']) > 0:
        for metric, data in breakdown['base_metrics'].items():
            if data['value'] is not None and data['threshold'] is not None:
                metrics_values.append(data['value'])
                metrics_thresholds.append(data['threshold'])
                metric_names.append(metric)
        
        if metric_names:  # Make sure we have data
            base_metrics_fig.add_trace(go.Scatterpolar(
                r=metrics_values,
                theta=metric_names,
                fill='toself',
                name='Current Values'
            ))
            
            base_metrics_fig.add_trace(go.Scatterpolar(
                r=metrics_thresholds,
                theta=metric_names,
                fill='toself',
                name='Industry Thresholds'
            ))
            
            base_metrics_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(metrics_values + metrics_thresholds) * 1.2])),
                showlegend=True,
                title='Base Metrics Analysis'
            )
            
            figures['base_metrics'] = base_metrics_fig
    
    # 2. Sector-Specific Metrics Bar Chart
    if breakdown['sector_specific']:
        sector_metrics_values = []
        sector_metrics_thresholds = []
        sector_metric_names = []
        
        for metric, data in breakdown['sector_specific'].items():
            if data['value'] is not None and data['threshold'] is not None:
                sector_metrics_values.append(data['value'])
                sector_metrics_thresholds.append(data['threshold'])
                sector_metric_names.append(metric)
        
        if sector_metric_names:  # Make sure we have data
            sector_fig = go.Figure(data=[
                go.Bar(name='Current Values', x=sector_metric_names, y=sector_metrics_values),
                go.Bar(name='Industry Thresholds', x=sector_metric_names, y=sector_metrics_thresholds)
            ])
            
            sector_fig.update_layout(
                barmode='group',
                title=f'{sector_type.value} Specific Metrics',
                xaxis_title='Metrics',
                yaxis_title='Values'
            )
            
            figures['sector_specific'] = sector_fig
    
    # 3. Risk Factors Gauge Chart
    if breakdown['risk_factors']:
        risk_values = list(breakdown['risk_factors'].values())
        risk_names = list(breakdown['risk_factors'].keys())
        
        if risk_names:  # Make sure we have data
            risk_fig = go.Figure()
            
            for i, (name, value) in enumerate(zip(risk_names, risk_values)):
                risk_fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    domain={'row': i, 'column': 0},
                    title={'text': name},
                    gauge={'axis': {'range': [0, 100]},
                           'steps': [
                               {'range': [0, 33], 'color': "lightgreen"},
                               {'range': [33, 66], 'color': "yellow"},
                               {'range': [66, 100], 'color': "red"}
                           ]}
                ))
            
            risk_fig.update_layout(
                grid={'rows': len(risk_names), 'columns': 1, 'pattern': "independent"},
                height=200 * len(risk_names),
                title='Risk Factor Analysis'
            )
            
            figures['risk_factors'] = risk_fig
    
    # 4. Composite Score Gauge
    composite_score = metrics.get('Valuation Score', 50)
    
    score_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=composite_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Score"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 20], 'color': "red"},
                {'range': [20, 40], 'color': "orange"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "lightgreen"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': composite_score
            }
        }
    ))
    
    score_fig.update_layout(height=400)
    figures['composite_score'] = score_fig
    
    # 5. Historical Performance (if available)
    if 'historical_data' in metrics:
        hist_fig = go.Figure()
        hist_data = metrics['historical_data']
        
        hist_fig.add_trace(go.Scatter(
            x=hist_data['dates'],
            y=hist_data['values'],
            mode='lines',
            name='Historical Performance'
        ))
        
        hist_fig.update_layout(
            title='Historical Performance Analysis',
            xaxis_title='Date',
            yaxis_title='Value'
        )
        
        figures['historical'] = hist_fig
    
    return figures

def color_recommendation(val):
    """Style function for recommendations"""
    colors = {
        'Strong Buy': 'background-color: #2E7D32; color: white',
        'Buy': 'background-color: #4CAF50; color: white',
        'Hold': 'background-color: #FFC107',
        'Sell': 'background-color: #F44336; color: white',
        'Strong Sell': 'background-color: #B71C1C; color: white'
    }
    return colors.get(val, '')

def display_stock_info(ticker, company_name, sector, exchange, price):
    """Show basic info when stock is added"""
    st.sidebar.markdown(f"""
    **Added: {company_name}**
    - Exchange: {exchange}
    - Sector: {sector}
    - Current Price: ${price:.2f}
    """)

def map_sector_to_type(sector):
    """Map FMP sectors to SectorType"""
    sector_map = {
        'Technology': SectorType.TECHNOLOGY,
        'Healthcare': SectorType.HEALTHCARE,
        'Financials': SectorType.FINANCIAL,
        'Financial': SectorType.FINANCIAL, 
        'Consumer Discretionary': SectorType.CONSUMER_CYCLICAL,
        'Consumer Staples': SectorType.CONSUMER_DEFENSIVE,
        'Industrials': SectorType.INDUSTRIALS,
        'Materials': SectorType.BASIC_MATERIALS,
        'Energy': SectorType.ENERGY,
        'Utilities': SectorType.UTILITIES,
        'Real Estate': SectorType.REAL_ESTATE,
        'Communication Services': SectorType.COMMUNICATION,
        'Telecommunications': SectorType.COMMUNICATION
    }
    return sector_map.get(sector, SectorType.TECHNOLOGY)