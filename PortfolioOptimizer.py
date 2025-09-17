'''
TITLE: Portfolio Optimizer
AUTHOR: JGStrickland
DESCRIPTION: Implementation of Modern Portfolio Theory (Markowitz model) with Monte Carlo simulation
optimized for efficient frontier analysis. The tool calculates optimal asset allocation by maximizing
the Sharpe ratio or minimizing variance, visualizes the efficient frontier, and provides detailed
portfolio statistics. Based on the work of Holczer Balazs from the Udemy course
"Quantitative Finance & Algorithmic Trading in Python".

KEY FEATURES:
- Historical data retrieval from Yahoo Finance
- Risk-return characteristics analysis
- Monte Carlo simulation for portfolio optimization
- Efficient frontier visualization
- Optimal portfolio allocation with Sharpe ratio maximization
- Statistical summary and covariance matrix visualization
'''

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import seaborn as sns
from datetime import datetime

# Global constants
NUM_TRADING_DAYS = 252  # Average number of trading days in a year
NUM_PORTFOLIOS = 50000  # Number of random portfolios to generate
RISK_FREE_RATE = 0.02   # Assumed risk-free rate for Sharpe ratio calculation

class PortfolioOptimizer:
    """
    A class to optimize portfolio allocation using Modern Portfolio Theory.

    Attributes:
        stocks (list): List of stock tickers
        start_date (str): Start date for historical data
        end_date (str): End date for historical data
        data (DataFrame): Historical price data
        returns (DataFrame): Logarithmic returns of the assets
        optimal_result (OptimizeResult): Results of portfolio optimization
    """

    def __init__(self, stocks, start_date, end_date):
        """
        Initialize the PortfolioOptimizer with stocks and date range.

        Args:
            stocks (list): List of stock ticker symbols
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        self.optimal_result = None

    def download_data(self):
        """
        Download historical stock price data from Yahoo Finance.

        Raises:
            ConnectionError: If data download fails
        """
        try:
            self.data = yf.download(self.stocks, start=self.start_date,
                                    end=self.end_date, auto_adjust=True)['Close'].dropna()
            print(f"Successfully downloaded data for {len(self.stocks)} assets")
        except Exception as e:
            raise ConnectionError(f"Failed to download data: {str(e)}")

    def calculate_returns(self):
        """
        Calculate logarithmic returns from price data.

        Returns:
            DataFrame: Daily logarithmic returns

        Raises:
            ValueError: If price data is not available
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call download_data() first.")

        # Calculate daily log returns (ln(Pt/Pt-1))
        self.returns = np.log(self.data / self.data.shift(1)).dropna()
        return self.returns

    def show_data_plot(self):
        """
        Plot normalized price evolution of all assets.

        Raises:
            ValueError: If data is not available
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call download_data() first.")

        # Normalize prices to compare performance from start date
        (self.data / self.data.iloc[0] * 100).plot(figsize=(12, 8))
        plt.title('Stock Price Evolution (Normalized to 100)')
        plt.ylabel('Normalised Price')
        plt.xlabel('Date')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def show_statistics(self):
        """
        Display statistical summary of returns including annualized returns,
        volatility, and covariance matrix heatmap.

        Raises:
            ValueError: If returns data is not calculated
        """
        if self.returns is None:
            self.calculate_returns()

        # Calculate and display annualized returns
        annual_returns = self.returns.mean() * NUM_TRADING_DAYS
        print("Annualised Returns (%):")
        print((annual_returns * 100).round(2))

        # Calculate and display annualized volatility
        annual_volatility = self.returns.std() * np.sqrt(NUM_TRADING_DAYS)
        print("\nAnnualised Volatility (%):")
        print((annual_volatility * 100).round(2))

        # Display covariance matrix as heatmap
        cov_matrix = self.returns.cov() * NUM_TRADING_DAYS
        plt.figure(figsize=(10, 8))
        sns.heatmap(cov_matrix.round(6), annot=True, cmap='coolwarm',
                    square=True, fmt='.6f', cbar_kws={'shrink': 0.8})
        plt.title('Annualized Covariance Matrix')
        plt.tight_layout()
        plt.show()

    def generate_random_portfolios(self):
        """
        Generate random portfolios for Monte Carlo simulation.

        Returns:
            tuple: Arrays of weights, returns, risks, and Sharpe ratios

        Raises:
            ValueError: If returns data is not calculated
        """
        if self.returns is None:
            self.calculate_returns()

        # Precompute annualized values for efficiency
        annual_returns = self.returns.mean() * NUM_TRADING_DAYS
        annual_cov = self.returns.cov() * NUM_TRADING_DAYS

        # Generate random portfolio weights
        weights = np.random.random((NUM_PORTFOLIOS, len(self.stocks)))
        weights /= weights.sum(axis=1)[:, np.newaxis]  # Normalize to sum to 1

        # Vectorized return and risk calculation
        portfolio_returns = np.dot(weights, annual_returns)
        portfolio_risks = np.sqrt(np.einsum('ij,ij->i', weights,
                                            np.dot(weights, annual_cov)))

        # Calculate Sharpe ratios (risk-adjusted returns)
        sharpe_ratios = (portfolio_returns - RISK_FREE_RATE) / portfolio_risks

        return weights, portfolio_returns, portfolio_risks, sharpe_ratios

    def portfolio_statistics(self, weights):
        """
        Calculate key statistics for a given portfolio allocation.

        Args:
            weights (array): Portfolio weights for each asset

        Returns:
            tuple: Portfolio return, risk, and Sharpe ratio

        Raises:
            ValueError: If returns data is not calculated
        """
        if self.returns is None:
            self.calculate_returns()

        annual_returns = self.returns.mean() * NUM_TRADING_DAYS
        annual_cov = self.returns.cov() * NUM_TRADING_DAYS

        # Calculate portfolio return and risk
        port_return = np.dot(weights, annual_returns)
        port_risk = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
        sharpe_ratio = (port_return - RISK_FREE_RATE) / port_risk

        return port_return, port_risk, sharpe_ratio

    def optimize_portfolio(self, objective='sharpe'):
        """
        Optimize portfolio allocation for a given objective function.

        Args:
            objective (str): 'sharpe' to maximize Sharpe ratio or
                            'min_variance' to minimize variance

        Returns:
            OptimizeResult: Optimization results including optimal weights

        Raises:
            ValueError: If returns data is not calculated or invalid objective
        """
        if self.returns is None:
            self.calculate_returns()

        annual_returns = self.returns.mean() * NUM_TRADING_DAYS
        annual_cov = self.returns.cov() * NUM_TRADING_DAYS

        # Define constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(len(self.stocks)))  # No short selling
        initial_guess = np.array([1 / len(self.stocks)] * len(self.stocks))  # Equal weights

        # Define objective function based on optimization goal
        if objective == 'sharpe':
            def objective_function(weights):
                port_return = np.dot(weights, annual_returns)
                port_risk = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
                return -(port_return - RISK_FREE_RATE) / port_risk  # Negative for minimization
        elif objective == 'min_variance':
            def objective_function(weights):
                return np.dot(weights.T, np.dot(annual_cov, weights))
        else:
            raise ValueError("Objective must be 'sharpe' or 'min_variance'")

        # Perform optimization using Sequential Least Squares Programming
        result = optimization.minimize(
            objective_function,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        self.optimal_result = result
        return result

    def plot_efficient_frontier(self, portfolio_returns, portfolio_risks, sharpe_ratios):
        """
        Plot the efficient frontier with optimal portfolio highlighted.

        Args:
            portfolio_returns (array): Returns of generated portfolios
            portfolio_risks (array): Risks of generated portfolios
            sharpe_ratios (array): Sharpe ratios of generated portfolios

        Raises:
            ValueError: If optimization hasn't been performed
        """
        if self.optimal_result is None:
            raise ValueError("No optimal portfolio found. Run optimize_portfolio() first.")

        plt.figure(figsize=(12, 8))
        # Create scatter plot of all simulated portfolios
        scatter = plt.scatter(portfolio_risks, portfolio_returns,
                              c=sharpe_ratios, cmap='viridis', marker='o',
                              alpha=0.7, s=15)
        plt.colorbar(scatter, label='Sharpe Ratio')

        # Plot the optimal portfolio
        opt_return, opt_risk, opt_sharpe = self.portfolio_statistics(self.optimal_result.x)
        plt.plot(opt_risk, opt_return, 'r*', markersize=20, label='Optimal Portfolio')

        # Add text annotation for optimal portfolio
        plt.annotate(f'Sharpe: {opt_sharpe:.2f}\nReturn: {opt_return * 100:.1f}%\nRisk: {opt_risk * 100:.1f}%',
                     xy=(opt_risk, opt_return), xytext=(50, 50), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3'))

        plt.title('Efficient Frontier and Random Portfolios')
        plt.xlabel('Annualized Risk (Standard Deviation)')
        plt.ylabel('Annualized Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_optimal_portfolio_allocation(self):
        """
        Plot pie chart of optimal portfolio allocation.

        Raises:
            ValueError: If optimization hasn't been performed
        """
        if self.optimal_result is None:
            raise ValueError("No optimal portfolio found. Run optimize_portfolio() first.")

        weights = self.optimal_result.x

        # Filter out assets with negligible weights (<1%)
        significant_weights = weights > 0.01
        plot_stocks = np.array(self.stocks)[significant_weights]
        plot_weights = weights[significant_weights]

        # Create pie chart
        plt.figure(figsize=(10, 8))
        patches, texts, autotexts = plt.pie(
            plot_weights,
            labels=plot_stocks,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Set3(np.linspace(0, 1, len(plot_stocks)))
        )

        # Improve text appearance
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.title('Optimal Portfolio Allocation')
        plt.axis('equal')  # Ensure pie is drawn as a circle
        plt.tight_layout()
        plt.show()

    def print_optimization_result(self):
        """
        Print formatted optimization results including weights and performance metrics.

        Raises:
            ValueError: If optimization hasn't been performed
        """
        if self.optimal_result is None:
            raise ValueError("No optimal portfolio found. Run optimize_portfolio() first.")

        weights = self.optimal_result.x
        return_val, risk, sharpe = self.portfolio_statistics(weights)

        print("=" * 60)
        print("PORTFOLIO OPTIMIZATION RESULTS")
        print("=" * 60)
        print(f"{'Stock':<15} {'Weight':>15} {'Annual Return':>15} {'Annual Volatility':>15}")
        print("-" * 60)

        # Print individual asset information
        annual_returns = self.returns.mean() * NUM_TRADING_DAYS
        annual_volatility = self.returns.std() * np.sqrt(NUM_TRADING_DAYS)

        for i, stock in enumerate(self.stocks):
            # Use .iloc for positional indexing to avoid the warning
            print(f"{stock:<15} {weights[i]:>15.2%} {annual_returns.iloc[i]:>15.2%} {annual_volatility.iloc[i]:>15.2%}")

        print("-" * 60)
        print(f"{'PORTFOLIO TOTAL':<15} {np.sum(weights):>15.2%} {return_val:>15.2%} {risk:>15.2%}")
        print("-" * 60)
        print(f"{'Sharpe Ratio':<30} {sharpe:>15.2f}")
        print(f"{'Risk-Free Rate':<30} {RISK_FREE_RATE:>15.2%}")
        print("=" * 60)

    def calculate_efficient_frontier(self, target_returns=None):
        """
        Calculate the efficient frontier for a range of target returns.

        Args:
            target_returns (array): Specific target returns to calculate frontier for

        Returns:
            tuple: Arrays of efficient risks and returns

        Raises:
            ValueError: If returns data is not calculated
        """
        if self.returns is None:
            self.calculate_returns()

        annual_returns = self.returns.mean() * NUM_TRADING_DAYS
        annual_cov = self.returns.cov() * NUM_TRADING_DAYS

        # If no target returns specified, create a range from min to max return
        if target_returns is None:
            min_return = annual_returns.min()
            max_return = annual_returns.max()
            target_returns = np.linspace(min_return, max_return, 50)

        efficient_risks = []

        for target_return in target_returns:
            # Define constraints for target return and weight sum
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, annual_returns) - target_return}
            ]
            bounds = tuple((0, 1) for _ in range(len(self.stocks)))
            initial_guess = np.array([1 / len(self.stocks)] * len(self.stocks))

            # Minimize risk for target return
            result = optimization.minimize(
                lambda x: np.sqrt(np.dot(x.T, np.dot(annual_cov, x))),
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                efficient_risks.append(result.fun)
            else:
                efficient_risks.append(np.nan)

        return np.array(efficient_risks), target_returns

def main():
    """
    Main function to demonstrate the portfolio optimization process.
    """
    # Configuration
    stocks = ['V', 'GE', 'ORCL', 'MSFT', 'JPM', 'JNJ', 'PG']
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')  # Use current date as end date

    # Initialize optimizer
    optimizer = PortfolioOptimizer(stocks, start_date, end_date)

    try:
        # Download and process data
        print("Downloading historical data...")
        optimizer.download_data()

        print("Calculating returns...")
        optimizer.calculate_returns()

        # Display data and statistics
        print("Generating price evolution plot...")
        optimizer.show_data_plot()

        print("Calculating statistics...")
        optimizer.show_statistics()

        # Generate random portfolios
        print(f"Generating {NUM_PORTFOLIOS} random portfolios...")
        weights, returns, risks, sharpes = optimizer.generate_random_portfolios()

        # Optimize portfolio
        print("Optimizing portfolio for maximum Sharpe ratio...")
        optimizer.optimize_portfolio(objective='sharpe')

        # Display results
        optimizer.print_optimization_result()

        print("Plotting efficient frontier...")
        optimizer.plot_efficient_frontier(returns, risks, sharpes)

        print("Plotting optimal allocation...")
        optimizer.plot_optimal_portfolio_allocation()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()