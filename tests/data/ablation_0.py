import os
import shutil
import glob

def create_ablation_folders():
    """
    Create ablation folders with different total character limits:
    - articles_40k_ch: 40,000 characters
    - articles_400k_ch: 400,000 characters
    - articles_4m_ch: 4,000,000 characters
    - articles_20m_ch: 20,000,000 characters
    
    This function copies article files from the source directory to each target directory
    until the specified character limit is reached. If including the next file would exceed
    the limit, the file is truncated to fit within the remaining character budget.
    
    The function creates a series of increasingly larger datasets that can be used for
    ablation studies to measure how knowledge graph generation performance scales with
    input data size.
    """
    # Define source directory and target directories with character limits
    source_dir = "tests/data/wiki_qa/articles"
    target_dirs = {
        "tests/data/wiki_qa/articles_40k_ch": 40_000,
        "tests/data/wiki_qa/articles_400k_ch": 400_000,
        "tests/data/wiki_qa/articles_4m_ch": 4_000_000,
        "tests/data/wiki_qa/articles_20m_ch": 20_000_000
    }
    
    # Create target directories if they don't exist
    for target_dir in target_dirs:
        os.makedirs(target_dir, exist_ok=True)
        print(f"Created directory: {target_dir}")
    
    # Get all article files in alphabetical order
    article_files = sorted(glob.glob(os.path.join(source_dir, "*.txt")))
    
    # Process each target directory with its character limit
    for target_dir, char_limit in target_dirs.items():
        total_chars = 0
        files_copied = 0
        
        print(f"\nProcessing {target_dir} (limit: {char_limit:,} characters)")
        
        for article_file in article_files:
            with open(article_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_chars = len(content)
            
            # If adding this file would exceed the limit
            if total_chars + file_chars > char_limit:
                # Calculate how many characters we can include from this file
                chars_remaining = char_limit - total_chars
                
                if chars_remaining > 0:
                    # Truncate the content and write to the target directory
                    truncated_content = content[:chars_remaining]
                    filename = os.path.basename(article_file)
                    target_path = os.path.join(target_dir, filename)
                    
                    with open(target_path, 'w', encoding='utf-8') as f:
                        f.write(truncated_content)
                    
                    print(f"Truncated {filename} to {chars_remaining:,} characters")
                    files_copied += 1
                    total_chars += chars_remaining
                
                break  # Stop processing more files for this directory
            
            # Copy the entire file
            filename = os.path.basename(article_file)
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(article_file, target_path)
            
            total_chars += file_chars
            files_copied += 1
        
        print(f"Completed {target_dir}: {files_copied} files, {total_chars:,}/{char_limit:,} characters")

if __name__ == "__main__":
    create_ablation_folders()
