from flask import Flask, render_template, request, jsonify
import plotly.express as px
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    # Load dataset with low_memory=False and specify dtype if necessary
    data = pd.read_csv('data/dataset.csv', low_memory=False)
    data['net_worth'] = data['net_worth'].str.replace(' B', '').astype(float)
    
    data['year'] = data['year'].astype(int)  # Ensure the year column is of integer type

    # Get unique years from the dataset and group by decade
    unique_years = sorted(data['year'].unique())
    print("Unique years in the dataset:", unique_years)  # Debugging print

    years = {}
    for year in unique_years:
        decade = (year // 10) * 10
        if decade not in years:
            years[decade] = []
        years[decade].append(year)

    scatter_fig = px.scatter(data, x='age', y='net_worth', color='gender', title='Net Worth by Age')
    scatter_html = scatter_fig.to_html(full_html=False)
    
    # Change bar graph to line graph
    line_fig = px.line(data, x='year', y='net_worth', color='gender', title='Net Worth by Year')
    line_html = line_fig.to_html(full_html=False)
    
    return render_template('index.html', scatter_html=scatter_html, line_html=line_html, years=years)

@app.route('/filter')
def filter_data():
    year = request.args.get('year')
    print("Year received in request:", year)  # Debugging print
    data = pd.read_csv('data/dataset.csv', low_memory=False)
    data['net_worth'] = data['net_worth'].str.replace(' B', '').astype(float)
    
    data['year'] = data['year'].astype(int)  # Ensure the year column is of integer type

    if year == "All":
        filtered_data = data
    else:
        filtered_data = data[data['year'] == int(year)]
    print("Filtered data for year {}:".format(year))  # Debugging print
    print(filtered_data.head())  # Debugging print

    scatter_fig = px.scatter(filtered_data, x='age', y='net_worth', color='gender', title=f'Net Worth by Age ({year})')
    scatter_json = scatter_fig.to_json()
    
    # Change bar graph to line graph
    line_fig = px.line(filtered_data, x='year', y='net_worth', color='gender', title=f'Net Worth by Year ({year})')
    line_json = line_fig.to_json()
    
    return jsonify(scatter_json=scatter_json, line_json=line_json)

if __name__ == '__main__':
    app.run(debug=True)
