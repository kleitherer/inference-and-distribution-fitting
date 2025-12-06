import csv

def get_bucket(pr):
    # return the bucket that pr is closest to
    # >>> get_bucket(0.23) will return 0.2
    # >>> get_bucket(0.55) will return 0.6
    buckets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    return min(buckets, key=lambda x: abs(x - pr))

file_path = '/Users/kaitlynleitherer/Desktop/CS109/pset7/ChatGPTPr.csv'

def load_csv(chat_buckets):
    with open(file_path, 'r') as file:
        csv_data = csv.reader(file)
        next(csv_data)  # Skip the header row

        rows = list(csv_data)
        # number of labels is the number of rows in the csv file
        num_labels = len(rows)
        count_06 = 0
        for row in rows:
            chat_pr = row[1]
            label = row[0]
            # convert to string to float
            chat_pr = float(str(chat_pr))
            bucket = get_bucket(chat_pr)
            if bucket not in chat_buckets:
                chat_buckets[bucket] = 0
            chat_buckets[bucket] += 1
            
            # count the fraction of times the label was 1 for 0.6
            if bucket == 0.6:
                if label == '1':
                    count_06 += 1

        return chat_buckets, num_labels, count_06

def main():
    chat_buckets = {}
    chat_pr_buckets, total_label_1, count_06 = load_csv(chat_buckets)
    print(chat_pr_buckets, total_label_1, count_06)
    # Then, out of all the points with ChatGPT_pr closest to 0.6 count the fraction of times the Label was 1. 
    print(count_06/chat_pr_buckets[0.6])


if __name__ == "__main__":
    main()

