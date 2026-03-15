---
name: "Workout Guidance"
description: "Comprehensive workout planning and advice"
version: "1.0.0"
tags: ["fitness", "workout", "planning"]
author: "user"
readonly: false
---

# Workout Guidance

## Purpose
Provide personalized workout advice including:
- Complete training plans
- Exercise selection and modifications
- Form guidance
- Progression strategies
- Specific workout questions

## Scope
This skill handles ALL workout-related requests:
- "Create a workout plan for me"
- "How do I do a proper squat?"
- "Should I train abs every day?"
- "What exercises for lower back pain?" (with safety caveats)

## Process

### 1. Load User Context
```
Always check: /memory/user_profiles/{user_id}.json
Required info:
- fitness_level (beginner/intermediate/advanced)
- goals (strength/hypertrophy/fat_loss/endurance)
- equipment_access (gym/home/minimal)
- schedule (days per week, time per session)
- limitations (injuries, conditions)
```

### 2. For Workout Plans

#### Choose Training Split
- **2-3 days/week**: Full body each session
- **4 days/week**: Upper/Lower or Push/Pull
- **5-6 days/week**: Push/Pull/Legs or Body Part Split

#### Exercise Selection Principles
1. Start with compounds: squat, deadlift, press, row, pull-up
2. Add isolation based on goals
3. Match equipment available
4. Consider limitations (e.g., no barbell = use dumbbells)

#### Volume Guidelines
**Beginner** (0-1 year training):
- 2-3 sets per exercise
- 10-15 total sets per muscle per week
- Focus on form and consistency

**Intermediate** (1-3 years):
- 3-4 sets per exercise
- 15-20 total sets per muscle per week
- Progressive overload each week

**Advanced** (3+ years):
- 4-6 sets per exercise
- 20-25 total sets per muscle per week
- Periodization (vary intensity/volume)

#### Rep Ranges by Goal
- **Strength**: 3-6 reps, 85-95% 1RM, 3-5min rest
- **Hypertrophy**: 8-12 reps, 70-80% 1RM, 60-90s rest
- **Endurance**: 15+ reps, 50-65% 1RM, 30-60s rest

### 3. For Specific Questions

#### Exercise Form
- Load the exercise from memory or search if needed
- Provide key cues (e.g., "chest up, knees out" for squats)
- Mention common mistakes
- Suggest lighter weight to practice form

#### Exercise Substitutions
- Barbell → Dumbbell (more accessible)
- Gym machine → Bodyweight/band alternative
- High-impact → Low-impact (for joint issues)

#### Progression Strategies
- Linear progression (add 5lbs per week) - beginners
- Double progression (increase reps, then weight) - intermediate
- Periodization (vary volume/intensity) - advanced

### 4. Research Integration
When you need updated information:
- Use `web_search` for: exercise variations, latest research, injury prevention
- Apply `fitness/safety` filtering to sources
- Cite sources in your response
- Cross-reference with established knowledge

### 5. Save Plans
```
Save to: /memory/plans/{user_id}/{timestamp}_{plan_name}.json

Include:
- Full workout structure
- Exercise details (sets/reps/rest)
- Safety notes
- Progression protocol
- Skills used (for tracking)
```

## Equipment Adaptations

### Home Gym (Dumbbells + Bench)
- Barbell bench → Dumbbell press
- Barbell squat → Goblet squat or Bulgarian split squat
- Barbell row → Dumbbell row
- Deadlift → Dumbbell RDL

### Minimal Equipment (Bodyweight)
- Focus on calisthenics progressions
- Push-ups → Archer push-ups → One-arm push-ups
- Pull-ups (if bar available) → Assisted → Weighted
- Squats → Pistol squat progression
- Add resistance bands for variety

## Common Scenarios

### Busy Schedule
- 30-min workouts: Focus on compounds, supersets
- 3 days/week max: Full body with 4-5 exercises
- Time-efficient: Drop sets, circuits

### Fat Loss Focus
- Prioritize compound movements (higher calorie burn)
- Add metabolic finishers (HIIT circuits)
- 3-5 days lifting + 2-3 days cardio
- Mention: Training doesn't outwork bad diet (refer to diet skill)

### Muscle Building
- 4-6 days per week
- Higher volume (15-20 sets per muscle)
- 8-12 rep range dominant
- Emphasize progressive overload

## Cross-References
- Always apply `fitness/safety` rules
- For nutrition questions → Load `fitness/diet` skill
- For research → Use `research/web_search` strategy

## Known Gaps (Ideas for Improvement)
- Need more specific mobility/stretching protocols
- Could add deload week guidelines
- Should include injury return-to-training progressions