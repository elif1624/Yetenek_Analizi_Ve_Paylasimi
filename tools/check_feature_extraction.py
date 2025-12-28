import json

# Dosyaları yükle
labels = json.load(open('data/labels/nba_test_video_labels.json', encoding='utf-8'))
analysis = json.load(open('data/results/nba_test_video_final_analysis.json', encoding='utf-8'))

# Analiz edilen frame'leri al
frame_results = {fr['frame_number']: fr for fr in analysis.get('frame_results', [])}

print(f'Analiz edilen frame sayisi: {len(frame_results)}')
print(f'Etiket sayisi: {len(labels["labels"])}')
print(f'\nFrame araligi: {min(frame_results.keys())} - {max(frame_results.keys())}')
print(f'\nİlk 10 etiket kontrolu:')
print('-' * 60)

for i, label in enumerate(labels['labels'][:10]):
    start, end = label['start_frame'], label['end_frame']
    frames_in_analysis = [f for f in range(start, end+1) if f in frame_results]
    frames_with_players = [f for f in frames_in_analysis if frame_results[f].get('tracked_players')]
    
    print(f'{i+1}. {label["event_type"].upper()}: Frame {start}-{end}')
    print(f'   Analizde var: {len(frames_in_analysis)}/{end-start+1} frame')
    print(f'   Oyuncu var: {len(frames_with_players)} frame')
    print()






