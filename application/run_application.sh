echo "Fitting model on real data..."
jupyter nbconvert --execute --to notebook --output out_application --output-dir ./out fit_real_data_8_chains.ipynb
echo "Model fitting is done! Generating results..."
jupyter nbconvert --execute --to notebook --output results_application --output-dir ./out analyze_application.ipynb
echo "Finished!"
