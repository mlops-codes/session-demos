#!/usr/bin/env python3
"""
Data Generator for SageMaker Feature Store Demo
Generates realistic customer and transaction data for feature store examples
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import random
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

class DataGenerator:
    """Generates realistic customer and transaction data"""
    
    def __init__(self, environment='development'):
        self.config = get_config(environment)
        np.random.seed(42)  # For reproducible results
        random.seed(42)
        
        # Data generation parameters
        self.customer_segments = ['Bronze', 'Silver', 'Gold', 'Platinum']
        self.transaction_types = ['Purchase', 'Withdrawal', 'Transfer', 'Payment']
        self.merchant_categories = [
            'Grocery', 'Gas Station', 'Restaurant', 'Online', 'ATM',
            'Department Store', 'Pharmacy', 'Entertainment', 'Travel', 'Other'
        ]
        
    def generate_customers(self):
        """Generate customer data"""
        print(f"👥 Generating {self.config.NUM_CUSTOMERS:,} customers...")
        
        customers = []
        
        for i in range(self.config.NUM_CUSTOMERS):
            customer_id = f"cust_{i+1:06d}"
            
            # Generate customer demographics
            age = np.random.normal(45, 15)
            age = max(18, min(80, int(age)))  # Clamp to reasonable range
            
            # Income correlated with age (somewhat)
            base_income = 30000 + (age - 18) * 1000 + np.random.normal(0, 20000)
            income = max(20000, base_income)
            
            # Credit score correlated with income and age
            base_credit = 300 + (income / 1000) * 0.3 + age * 2
            credit_score = int(max(300, min(850, base_credit + np.random.normal(0, 50))))
            
            # Account age
            account_age_days = np.random.randint(30, 3650)  # 1 month to 10 years
            
            # Number of accounts (correlated with income)
            num_accounts = max(1, int(np.random.poisson(income / 50000) + 1))
            
            # Average monthly balance
            avg_monthly_balance = income / 12 * np.random.uniform(0.1, 2.0)
            
            # Customer segment based on income and balance
            if income > 100000 and avg_monthly_balance > 10000:
                segment = 'Platinum'
                is_premium = 1
            elif income > 75000 and avg_monthly_balance > 5000:
                segment = 'Gold'
                is_premium = 1
            elif income > 50000:
                segment = 'Silver'
                is_premium = 0
            else:
                segment = 'Bronze'
                is_premium = 0
            
            # Last login
            last_login_days = np.random.exponential(7)  # Most customers login recently
            last_login_days = min(365, int(last_login_days))
            
            # Event time (current timestamp)
            event_time = datetime.now().timestamp()
            
            customer = {
                'customer_id': customer_id,
                'age': age,
                'income': round(income, 2),
                'credit_score': credit_score,
                'account_age_days': account_age_days,
                'num_accounts': num_accounts,
                'avg_monthly_balance': round(avg_monthly_balance, 2),
                'customer_segment': segment,
                'is_premium': is_premium,
                'last_login_days': last_login_days,
                'event_time': event_time
            }
            
            customers.append(customer)
        
        return pd.DataFrame(customers)
    
    def generate_transactions(self, customers_df):
        """Generate transaction data"""
        print(f"💳 Generating {self.config.NUM_TRANSACTIONS:,} transactions...")
        
        transactions = []
        customer_ids = customers_df['customer_id'].tolist()
        
        # Create customer lookup for features
        customer_lookup = customers_df.set_index('customer_id').to_dict('index')
        
        # Generate transaction history over time
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.DATE_RANGE_DAYS)
        
        for i in range(self.config.NUM_TRANSACTIONS):
            transaction_id = f"txn_{i+1:08d}"
            customer_id = random.choice(customer_ids)
            customer_info = customer_lookup[customer_id]
            
            # Transaction timing
            transaction_date = start_date + timedelta(
                days=random.uniform(0, self.config.DATE_RANGE_DAYS)
            )
            hour_of_day = transaction_date.hour
            is_weekend = 1 if transaction_date.weekday() >= 5 else 0
            
            # Transaction amount (influenced by customer segment)
            segment = customer_info['customer_segment']
            if segment == 'Platinum':
                base_amount = np.random.lognormal(mean=5, sigma=1)
            elif segment == 'Gold':
                base_amount = np.random.lognormal(mean=4.5, sigma=0.8)
            elif segment == 'Silver':
                base_amount = np.random.lognormal(mean=4, sigma=0.8)
            else:  # Bronze
                base_amount = np.random.lognormal(mean=3.5, sigma=0.7)
            
            amount = round(max(5, base_amount), 2)
            
            # Transaction type
            transaction_type = random.choice(self.transaction_types)
            
            # Merchant category
            merchant_category = random.choice(self.merchant_categories)
            
            # International transaction (premium customers more likely)
            is_international = 1 if random.random() < (0.1 if customer_info['is_premium'] else 0.02) else 0
            
            # Days since last transaction (simplified - assume uniform distribution)
            days_since_last = np.random.exponential(3)
            days_since_last = min(30, int(days_since_last))
            
            # Amount vs average ratio (will be calculated properly in aggregation)
            # For now, use a placeholder that makes sense
            customer_avg = customer_info['avg_monthly_balance'] / 30  # rough daily spending
            amount_vs_avg_ratio = amount / max(customer_avg, 1)
            
            transaction = {
                'transaction_id': transaction_id,
                'customer_id': customer_id,
                'amount': amount,
                'transaction_type': transaction_type,
                'merchant_category': merchant_category,
                'is_weekend': is_weekend,
                'hour_of_day': hour_of_day,
                'days_since_last_transaction': days_since_last,
                'amount_vs_avg_ratio': round(amount_vs_avg_ratio, 3),
                'is_international': is_international,
                'event_time': transaction_date.timestamp()
            }
            
            transactions.append(transaction)
        
        return pd.DataFrame(transactions)
    
    def generate_aggregated_features(self, customers_df, transactions_df):
        """Generate aggregated customer features from transactions"""
        print("📊 Generating aggregated features...")
        
        # Calculate 30-day rolling features for each customer
        aggregated_features = []
        
        for _, customer in customers_df.iterrows():
            customer_id = customer['customer_id']
            
            # Filter transactions for this customer
            customer_txns = transactions_df[transactions_df['customer_id'] == customer_id]
            
            if len(customer_txns) == 0:
                # Customer with no transactions
                aggregated = {
                    'customer_id': customer_id,
                    'total_transactions_30d': 0,
                    'total_amount_30d': 0.0,
                    'avg_transaction_amount_30d': 0.0,
                    'unique_merchants_30d': 0,
                    'weekend_transaction_ratio_30d': 0.0,
                    'international_transaction_ratio_30d': 0.0,
                    'max_transaction_amount_30d': 0.0,
                    'transaction_frequency_score': 0.0,
                    'risk_score': 0.1,  # Low risk for inactive customers
                    'event_time': datetime.now().timestamp()
                }
            else:
                # Calculate aggregated features
                total_transactions = len(customer_txns)
                total_amount = customer_txns['amount'].sum()
                avg_amount = customer_txns['amount'].mean()
                unique_merchants = customer_txns['merchant_category'].nunique()
                weekend_ratio = customer_txns['is_weekend'].mean()
                international_ratio = customer_txns['is_international'].mean()
                max_amount = customer_txns['amount'].max()
                
                # Transaction frequency score (transactions per day)
                frequency_score = total_transactions / 30.0
                
                # Risk score calculation (simplified)
                risk_factors = [
                    international_ratio * 0.3,  # International transactions
                    min(avg_amount / 1000, 1) * 0.2,  # High amounts
                    min(frequency_score / 5, 1) * 0.2,  # High frequency
                    (customer_txns['hour_of_day'] < 6).mean() * 0.2,  # Late night transactions
                    weekend_ratio * 0.1  # Weekend activity
                ]
                risk_score = sum(risk_factors)
                
                aggregated = {
                    'customer_id': customer_id,
                    'total_transactions_30d': total_transactions,
                    'total_amount_30d': round(total_amount, 2),
                    'avg_transaction_amount_30d': round(avg_amount, 2),
                    'unique_merchants_30d': unique_merchants,
                    'weekend_transaction_ratio_30d': round(weekend_ratio, 3),
                    'international_transaction_ratio_30d': round(international_ratio, 3),
                    'max_transaction_amount_30d': round(max_amount, 2),
                    'transaction_frequency_score': round(frequency_score, 3),
                    'risk_score': round(min(risk_score, 1.0), 3),
                    'event_time': datetime.now().timestamp()
                }
            
            aggregated_features.append(aggregated)
        
        return pd.DataFrame(aggregated_features)
    
    def generate_all_data(self):
        """Generate all datasets"""
        print("🏭 Generating all feature datasets...")
        
        # Generate base data
        customers_df = self.generate_customers()
        transactions_df = self.generate_transactions(customers_df)
        aggregated_df = self.generate_aggregated_features(customers_df, transactions_df)
        
        # Add some data quality insights
        print(f"\n📊 Data Summary:")
        print(f"   Customers: {len(customers_df):,} records")
        print(f"   Transactions: {len(transactions_df):,} records")
        print(f"   Aggregated: {len(aggregated_df):,} records")
        
        print(f"\n💰 Transaction Statistics:")
        print(f"   Total Amount: ${transactions_df['amount'].sum():,.2f}")
        print(f"   Average Amount: ${transactions_df['amount'].mean():.2f}")
        print(f"   International Rate: {transactions_df['is_international'].mean():.1%}")
        
        print(f"\n👥 Customer Statistics:")
        print(f"   Average Age: {customers_df['age'].mean():.1f} years")
        print(f"   Average Income: ${customers_df['income'].mean():,.0f}")
        print(f"   Premium Customers: {customers_df['is_premium'].mean():.1%}")
        
        return customers_df, transactions_df, aggregated_df
    
    def save_datasets(self, customers_df, transactions_df, aggregated_df, 
                     output_dir='data'):
        """Save datasets to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"💾 Saving datasets to {output_dir}/...")
        
        # Save as CSV and Parquet
        customers_df.to_csv(f'{output_dir}/customers.csv', index=False)
        customers_df.to_parquet(f'{output_dir}/customers.parquet', index=False)
        
        transactions_df.to_csv(f'{output_dir}/transactions.csv', index=False)
        transactions_df.to_parquet(f'{output_dir}/transactions.parquet', index=False)
        
        aggregated_df.to_csv(f'{output_dir}/aggregated.csv', index=False)
        aggregated_df.to_parquet(f'{output_dir}/aggregated.parquet', index=False)
        
        print(f"✅ Datasets saved successfully")
        
        return {
            'customers': f'{output_dir}/customers.parquet',
            'transactions': f'{output_dir}/transactions.parquet',
            'aggregated': f'{output_dir}/aggregated.parquet'
        }

def main():
    """Main function to generate data"""
    print("🎲 SageMaker Feature Store Data Generator")
    print("=" * 45)
    
    # Initialize generator
    environment = os.getenv('ENVIRONMENT', 'development')
    generator = DataGenerator(environment)
    
    try:
        # Generate all data
        customers_df, transactions_df, aggregated_df = generator.generate_all_data()
        
        # Save datasets
        file_paths = generator.save_datasets(customers_df, transactions_df, aggregated_df)
        
        print(f"\n🎉 Data generation completed successfully!")
        print(f"\n📁 Generated files:")
        for dataset, path in file_paths.items():
            print(f"   • {dataset}: {path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)