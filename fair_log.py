import pandas as pd
import os
import glob
from multiprocessing import Pool
import logging


class csvLogger:
    def __init__(self, filename, path='results'):
        self.count = 1
        self.filename = filename
        self.path = path

        self.__check_path__()

    def __check_path__(self):
        isExist = os.path.exists(self.path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.path)

    def __call__(self, named_dict):
        df = pd.DataFrame(named_dict)
        df.to_csv(f"./{self.path}/{self.filename}.csv", mode='a')


def read_csv_file(file_path: str) -> pd.DataFrame:
    """
    Read a single CSV file and return its DataFrame.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the CSV data
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        return pd.DataFrame()
    

def aggregate_csv_files(folder_path: str, output_file: str = 'aggregated_data.csv', 
                       num_processes: int = 10) -> None:
    """
    Aggregate multiple CSV files from a folder into a single CSV file using multiprocessing.
    Removes duplicate rows and deletes source files after successful aggregation.
    
    Args:
        folder_path (str): Path to the folder containing CSV files
        output_file (str): Name of the output CSV file (default: 'aggregated_data.csv')
        num_processes (int): Number of processes to use (default: None, uses CPU count)
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Get list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {folder_path}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Create a pool of workers
    if num_processes is None:
        num_processes = os.cpu_count()
    
    try:
        # Read all CSV files in parallel
        with Pool(processes=num_processes) as pool:
            dataframes = pool.map(read_csv_file, csv_files)
        
        # Filter out empty DataFrames (from failed reads)
        dataframes = [df for df in dataframes if not df.empty]
        
        if not dataframes:
            logger.error("No valid data found in CSV files")
            return
        
        # Concatenate all DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Remove duplicate rows
        # initial_rows = len(combined_df)
        # combined_df.drop_duplicates(inplace=True)
        # dropped_rows = initial_rows - len(combined_df)
        
        # Save the aggregated data
        combined_df.to_csv(output_file, index=False)
        
        logger.info(f"Successfully aggregated {len(csv_files)} files")
        # logger.info(f"Removed {dropped_rows} duplicate rows")
        logger.info(f"Final dataset has {len(combined_df)} rows")
        
        # Delete original CSV files
        for file_path in csv_files:
            try:
                os.remove(file_path)
                logger.debug(f"Deleted: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error during aggregation: {str(e)}")
        raise

if __name__ == '__main__':
    # Example usage
    folder_path = './results/'
    output_file = './results/aggregated_data.csv'
    aggregate_csv_files(folder_path, output_file)