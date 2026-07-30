import pandas as pd
import random

def build_elaborated_language_rows():
    scenarios = {
        "psychology": {
            "phrases": [
                "i am having a complete nervous breakdown over school work",
                "i feel like i am about to collapse from academic stress",
                "i am completely panicking about my upcoming midterm exam",
                "i feel super anxious and overwhelmed by my poor grades and low gpa",
                "i am stressed about failing this final presentation and viva tomorrow",
                "this massive assignment deadline is giving me severe test anxiety and task paralysis",
                "i cannot focus on my studies because i feel like an academic imposter who will fail",
                "cramming for this competitive quiz ranking is causing heavy cognitive overload",
                "i am obsessed with my cgpa calculation and perfectionist expectations are killing me",
                "scared of freezing up during my university examinations and ruining my career scores",
                "i am hitting a total wall with my thesis and my mind is blanking out completely"
            ],
            "titles": ["Academic Evaluation Anxiety", "Performance Anxiety Regulation", "Cognitive Overload Management", "GPA Evaluation Threat Reframe", "Acute Academic Burnout Collapse"],
            "insights": [
                "Tying personal human value to numerical GPA metrics triggers severe evolutionary threat responses.",
                "High-stakes academic evaluation creates cognitive interference, temporarily lowering working memory capacity.",
                "Perceived academic failure activates catastrophic thinking loops regarding long-term career viability."
            ],
            "exercises": [
                "Cognitive Reframing: Intentionally separate your human value from your test score snapshot.",
                "Somatic Grounding: Use the 5-4-3-2-1 technique. Force visual focus onto 5 distinct room objects.",
                "Time Pacing: Apply the strict 25-minute Pomodoro focus block followed by a 5-minute break."
            ]
        },
        "mental_health": {
            "phrases": [
                "i am having a severe nervous breakdown and mental collapse right now",
                "i am at my absolute breaking point and feel like i am losing my mind",
                "i cannot sleep at all because my mind is racing with bad nightmares and insomnia",
                "i feel completely exhausted tired and drained from chronic burnout and fatigue",
                "my chest is tight and i am hyperventilating shaking and panicking right now",
                "i feel deeply depressed sad unmotivated and empty inside my room",
                "having an acute panic attack and my heart is racing so fast it scares me",
                "i feel completely numb isolated from everyone and i keep crying for no reason",
                "brain fog is making it impossible to stay awake and i have zero energy left",
                "i am physically collapsing from stress and i cannot cope with reality anymore",
                "everything is spinning and i feel completely detached from my physical body"
            ],
            "titles": ["Somatic Hyper-Arousal Mitigation", "Chronic Burnout Energy Balancing", "Acute Panic Attack Somatic Override", "Nervous System De-escalation Protocol", "Severe Overwhelm Crisis Stabilization"],
            "insights": [
                "A nervous breakdown or feeling of collapse is your sympathetic nervous system firing an emergency shut-down signal due to prolonged, unmanaged hyper-arousal.",
                "Somatic hyper-arousal signals a false survival panic loop within your nervous pathways.",
                "Chronic emotional and energetic burnout manifests when cognitive output outpaces regular restorative breaks."
            ],
            "exercises": [
                "Immediate Neural Brake: Drop your shoulders, unclench your jaw, and lengthen your exhale to 6 seconds to manually force your vagus nerve to slow your heart rate.",
                "Somatic Reset: Use Box Breathing. Inhale smoothly for 4 seconds, hold for 4, exhale for 4, and hold empty for 4.",
                "Progressive Muscle Relaxation (PMR): Tense your foot muscles for 5 seconds, release completely, then work upward through your limbs."
            ]
        },
        "relationships": {
            "phrases": [
                "i just went through a terrible breakup and i feel completely lonely and isolated",
                "had a massive toxic fight and argument with my boyfriend and he is ghosting me",
                "my girlfriend lied to me and i have severe trust issues and jealousy problems",
                "i feel rejected by my close friends group and the distance is making me sad",
                "dealing with extreme family friction and my parents are constantly judging me",
                "misunderstanding with a friend has created a massive emotional distance between us",
                "i am feeling left out ghosted and rejected by someone i deeply care about",
                "caught in a codependent relationship cycle that is destroying my individual independence",
                "the loneliness is driving me to a complete breakdown and i feel totally abandoned"
            ],
            "titles": ["Relational Attachment Anxiety Management", "Interpersonal Conflict Resolution", "Situational Isolation Intervention", "Communication Disconnection Reframe", "Relational Rejection Processing"],
            "insights": [
                "Interpersonal conflict destabilizes primary social safety nets, immediately activating attachment panic.",
                "Situational loneliness in college ecosystems is a natural transitional phase, not a permanent reflection of your sociability.",
                "Defensive posturing during an argument stems from an internal need to protect oneself from perceived emotional rejection."
            ],
            "exercises": [
                "Assertive Communication: Deploy strict 'I' statements. Reframe defensive remarks into: 'I feel disconnected right now.'",
                "Relational Timeout: Mutually agree to a conditional 10-minute pause if emotional dysregulation peaks.",
                "Core Fact Isolation: Write down the raw, objective facts of your relationship argument on paper, separating them from emotional theories."
            ]
        }
    }

    domains = list(scenarios.keys())
    rows = []
    target_rows = 3000

    print(f"Generating {target_rows} highly elaborated conversational table rows...")

    for i in range(target_rows):
        dom = domains[i % len(domains)]
        pool = scenarios[dom]
        
        # Pull up to 3 distinct phrases to build a complex, varied user query string per row
        k_count = min(3, len(pool["phrases"]))
        selected_phrases = random.sample(pool["phrases"], k=k_count)
        conversational_sentence = " or ".join(selected_phrases)
        
        title_base = random.choice(pool["titles"])
        insight = random.choice(pool["insights"])
        ex1 = random.choice(pool["exercises"])
        ex2 = random.choice([e for e in pool["exercises"] if e != ex1])
        
        unique_title = f"{title_base} (Ref: P-{i+1})"
        formatted_answer = f"-> Clinical Insight: {insight}\n-> Psychological Exercise: {ex1}\n-> Actionable Next Step: {ex2}"
        
        rows.append({
            "domain": dom,
            "keyword_combinations": conversational_sentence,
            "predefined_title": unique_title,
            "predefined_answer": formatted_answer,
            "severity_tier": 2 if any(w in conversational_sentence for w in ["breakdown", "collapse", "breaking point", "panic"]) else 1,
            "action_url": f"https://university.edu/support/{dom}"
        })

    df = pd.DataFrame(rows)
    df.to_csv("your_3000_mental_health_matrix.csv", index=False)
    print("Success! Created expanded 'your_3000_mental_health_matrix.csv'.")

if __name__ == "__main__":
    build_elaborated_language_rows()