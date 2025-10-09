# Raw Data Directory

Place your raw survey data files here.

## Expected File Format

The main data file should be named `stress_survey_data.csv` and should include:

### Required Columns:
- `student_id` - Unique identifier for each student
- `meditation_practice` - Binary variable (0 = no meditation, 1 = practices meditation)
- `stress_score` - Primary outcome variable (e.g., PSS-10 score)

### Recommended Additional Columns:
- **Demographics**: age, gender, year_of_study
- **Academic**: gpa, study_hours_weekly
- **Mental Health**: anxiety_score, depression_score
- **Lifestyle**: sleep_hours, exercise_frequency
- **Social**: social_support_score
- **Qualitative**: open_ended_response (optional text field)

## Example Data Structure

```csv
student_id,meditation_practice,age,gender,year_of_study,gpa,study_hours_weekly,stress_score,anxiety_score,depression_score,sleep_hours,exercise_frequency,social_support_score,open_ended_response
1,1,20,Female,2,3.5,35,22,8,6,7.5,4,4,"Meditation has helped me manage stress..."
2,0,19,Male,1,3.2,40,28,12,10,6.0,2,3,"I find it difficult to balance academics..."
...
```

## Notes

- If no data file is provided, the scripts will automatically generate sample data for demonstration
- Ensure all sensitive data is properly anonymized before analysis
- Review data quality and completeness before running analyses
