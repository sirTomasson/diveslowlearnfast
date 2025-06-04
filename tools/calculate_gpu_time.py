from datetime import datetime


def parse_and_calculate_duration(start_str, end_str):
    """
    Parse datetime strings and calculate the time difference between them.

    Args:
        start_str (str): Start datetime string in format "Day Mon DD HH:MM:SS TIMEZONE YYYY"
        end_str (str): End datetime string in format "Day Mon DD HH:MM:SS TIMEZONE YYYY"

    Returns:
        dict: Dictionary containing parsed datetimes and duration information
    """

    # Define the format for parsing the datetime strings
    datetime_format = "%a %b %d %H:%M:%S %Z %Y"

    try:
        # Parse the datetime strings
        start_datetime = datetime.strptime(start_str, datetime_format)
        end_datetime = datetime.strptime(end_str, datetime_format)

        # Calculate the time difference
        duration = end_datetime - start_datetime

        # Extract duration components
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return {
            'start_datetime': start_datetime,
            'end_datetime': end_datetime,
            'duration': duration,
            'total_seconds': total_seconds,
            'total_minutes': round(total_seconds / 60, 2),
            'total_hours': round(total_seconds / 3600, 2),
            'formatted_duration': f"{hours}h {minutes}m {seconds}s"
        }

    except ValueError as e:
        return {'error': f"Error parsing datetime: {e}"}


# Your specific datetime strings
start_time = "Sun May 11 11:27:12 CEST 2025"
end_time = "Mon May 12 05:20:38 CEST 2025"

# Calculate the duration
result = parse_and_calculate_duration(start_time, end_time)

if 'error' not in result:
    print("Datetime Parsing and Duration Calculation")
    print("=" * 50)
    print(f"Start time: {result['start_datetime']}")
    print(f"End time:   {result['end_datetime']}")
    print(f"Duration:   {result['duration']}")
    print(f"Formatted:  {result['formatted_duration']}")
    print(f"Total hours: {result['total_hours']}")
    print(f"Total minutes: {result['total_minutes']}")
    print(f"Total seconds: {result['total_seconds']}")
else:
    print(f"Error: {result['error']}")


# Example with custom datetime strings
def calculate_custom_duration(start_str, end_str):
    """Convenience function for calculating duration between two datetime strings"""
    result = parse_and_calculate_duration(start_str, end_str)

    if 'error' not in result:
        print(f"\nDuration between:")
        print(f"  {start_str}")
        print(f"  {end_str}")
        print(f"Result: {result['formatted_duration']} ({result['total_hours']} hours)")
    else:
        print(f"Error: {result['error']}")


# Test with your specific dates
calculate_custom_duration(start_time, end_time)