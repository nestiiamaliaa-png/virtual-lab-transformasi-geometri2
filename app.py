# Virtual Lab: Transformasi Geometri (Rotasi, Dilatasi, Refleksi, Translasi)
# File: virtual_lab_transformasi_geometri_streamlit.py
# Deskripsi: Aplikasi Streamlit interaktif untuk memvisualisasikan transformasi geometri pada
# berbagai bentuk (segi tiga, persegi, poligon). Siswa dapat mengubah parameter transformasi,
# melihat matriks transformasi, animasi langkah demi langkah, serta mengunduh data koordinat.

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import base64

st.set_page_config(page_title="Virtual Lab: Transformasi Geometri", layout="wide")

# ---------------------- Helper functions ----------------------

def make_grid(xmin=-10, xmax=10, ymin=-10, ymax=10, step=1):
    xs = np.arange(xmin, xmax + step, step)
    ys = np.arange(ymin, ymax + step, step)
    return xs, ys


def apply_translation(points, tx, ty):
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    hom = np.vstack((points.T, np.ones(points.shape[0])))
    transformed = (T @ hom).T[:, :2]
    return transformed, T


def apply_rotation(points, angle_deg, origin=(0, 0)):
    theta = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    ox, oy = origin
    # Move to origin, rotate, move back: use homogeneous matrix
    T1 = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]])
    R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    T2 = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]])
    M = T2 @ R @ T1
    hom = np.vstack((points.T, np.ones(points.shape[0])))
    transformed = (M @ hom).T[:, :2]
    return transformed, M


def apply_scaling(points, sx, sy, origin=(0, 0)):
    ox, oy = origin
    T1 = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]])
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
    T2 = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]])
    M = T2 @ S @ T1
    hom = np.vstack((points.T, np.ones(points.shape[0])))
    transformed = (M @ hom).T[:, :2]
    return transformed, M


def apply_reflection(points, axis='x'):
    if axis == 'x':
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif axis == 'y':
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif axis == 'y=x':
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    elif axis == 'origin':
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        raise ValueError('Unknown reflection axis')
    hom = np.vstack((points.T, np.ones(points.shape[0])))
    transformed = (R @ hom).T[:, :2]
    return transformed, R


def chain_matrices(matrices):
    if not matrices:
        return np.eye(3)
    M = matrices[0]
    for mat in matrices[1:]:
        M = mat @ M
    return M


def polygon_examples(choice):
    if choice == "Segitiga (kanonik)":
        return np.array([[0, 0], [3, 0], [1.2, 2.5]])
    if choice == "Persegi":
        return np.array([[0, 0], [3, 0], [3, 3], [0, 3]])
    if choice == "Belah Ketupat":
        return np.array([[0, 2], [2, 0], [4, 2], [2, 4]])
    if choice == "Poligon Acak 6 sisi":
        return np.array([[0, 0], [2, -1], [4, 0.8], [3, 2.6], [1.5, 3.2], [-0.5, 1.5]])
    return np.array([[0,0],[1,0],[1,1],[0,1]])


def df_from_points(name, pts):
    return pd.DataFrame({ 'label': [f'{name}_{i}' for i in range(len(pts))], 'x': pts[:,0], 'y': pts[:,1] })


def points_to_csv_bytes(df):
    b = df.to_csv(index=False).encode('utf-8')
    return b


def plot_shapes(original, transformed, grid=True, title=''):
    # Boundaries
    allx = np.concatenate((original[:,0], transformed[:,0]))
    ally = np.concatenate((original[:,1], transformed[:,1]))
    margin = 1.5
    xmin, xmax = allx.min()-margin, allx.max()+margin
    ymin, ymax = ally.min()-margin, ally.max()+margin

    fig = go.Figure()

    # Grid lines
    if grid:
        xs, ys = make_grid(int(np.floor(xmin)), int(np.ceil(xmax)), int(np.floor(ymin)), int(np.ceil(ymax)), 1)
        for x in xs:
            fig.add_shape(type='line', x0=x, y0=ymin, x1=x, y1=ymax, line=dict(width=1, dash='dash'))
        for y in ys:
            fig.add_shape(type='line', x0=xmin, y0=y, x1=xmax, y1=y, line=dict(width=1, dash='dash'))

    # Original polygon
    x_o = np.append(original[:,0], original[0,0])
    y_o = np.append(original[:,1], original[0,1])
    fig.add_trace(go.Scatter(x=x_o, y=y_o, mode='lines+markers+text', name='Original', text=[f'{i}' for i in range(len(original))], textposition='top center', hoverinfo='text+x+y'))

    # Transformed polygon
    x_t = np.append(transformed[:,0], transformed[0,0])
    y_t = np.append(transformed[:,1], transformed[0,1])
    fig.add_trace(go.Scatter(x=x_t, y=y_t, mode='lines+markers+text', name='Transformed', text=[f'{i}' for i in range(len(transformed))], textposition='bottom center', hoverinfo='text+x+y'))

    # Connect corresponding vertices with faint lines
    for i in range(len(original)):
        fig.add_trace(go.Scatter(x=[original[i,0], transformed[i,0]], y=[original[i,1], transformed[i,1]], mode='lines', showlegend=False, line=dict(dash='dot')))

    fig.update_layout(height=600, width=800, xaxis=dict(range=[xmin, xmax], zeroline=True), yaxis=dict(range=[ymin, ymax], scaleanchor='x', scaleratio=1, zeroline=True), title=title)
    fig.update_yaxes(autorange=False)
    return fig

# ---------------------- Streamlit UI ----------------------

st.title('🎲 Virtual Lab: Transformasi Geometri')
st.write('Pelajari Rotasi, Dilatasi (Skala), Refleksi, dan Translasi secara interaktif. Ubah parameter, lihat matriks transformasi, dan unduh hasil koordinat.')

# Sidebar controls
with st.sidebar:
    st.header('Pengaturan Bentuk & Transformasi')
    shape_choice = st.selectbox('Pilih bentuk awal', ['Segitiga (kanonik)', 'Persegi', 'Belah Ketupat', 'Poligon Acak 6 sisi'])
    custom_coords = st.checkbox('Gunakan koordinat kustom (edit di bawah)', value=False)

    st.markdown('---')
    st.subheader('Transformasi (urut dari atas ke bawah)')
    translate_on = st.checkbox('Translasi', value=True)
    tx = st.number_input('tx', value=0.0, step=0.5) if translate_on else 0.0
    ty = st.number_input('ty', value=0.0, step=0.5) if translate_on else 0.0

    rotate_on = st.checkbox('Rotasi', value=True)
    angle = st.slider('Sudut rotasi (derajat)', -360, 360, 30) if rotate_on else 0
    rot_origin_choice = st.selectbox('Pusat rotasi', ['(0,0)', 'Centroid bentuk', 'Tentukan manual'])
    if rot_origin_choice == 'Tentukan manual':
        rot_ox = st.number_input('rot_ox', value=0.0, step=0.5)
        rot_oy = st.number_input('rot_oy', value=0.0, step=0.5)
    else:
        rot_ox = rot_oy = None

    scale_on = st.checkbox('Dilatasi (Skala)', value=True)
    sx = st.slider('sx', -3.0, 3.0, 1.0, step=0.1) if scale_on else 1.0
    sy = st.slider('sy', -3.0, 3.0, 1.0, step=0.1) if scale_on else 1.0
    scale_origin_choice = st.selectbox('Pusat dilatasi', ['(0,0)', 'Centroid bentuk', 'Tentukan manual'])
    if scale_origin_choice == 'Tentukan manual':
        scale_ox = st.number_input('scale_ox', value=0.0, step=0.5)
        scale_oy = st.number_input('scale_oy', value=0.0, step=0.5)
    else:
        scale_ox = scale_oy = None

    reflect_on = st.checkbox('Refleksi', value=False)
    reflect_axis = st.selectbox('Sumbu refleksi', ['x', 'y', 'y=x', 'origin']) if reflect_on else 'x'

    combine = st.checkbox('Gabungkan semua transformasi dalam satu komposisi (lihat matriks gabungan)', value=True)
    step_by_step = st.checkbox('Tampilkan langkah demi langkah', value=False)

    st.markdown('---')
    st.write('Aksi:')
    reset_button = st.button('Reset parameter ke default')
    export_csv = st.button('Unduh koordinat hasil (.csv)')

# Body layout
col1, col2 = st.columns([1, 1])

# Prepare polygon
if custom_coords:
    st.write('Masukkan koordinat bentuk sebagai list of tuples, contoh: [[0,0],[3,0],[1.2,2.5]]')
    coords_text = st.text_area('Koordinat (json-like)', value=str(polygon_examples(shape_choice)))
    try:
        pts = np.array(eval(coords_text))
    except Exception as e:
        st.error('Format koordinat tidak valid. Menggunakan bentuk default.')
        pts = polygon_examples(shape_choice)
else:
    pts = polygon_examples(shape_choice)

# Compute centroid
centroid = pts.mean(axis=0)

# Origin choices
if rot_origin_choice == 'Centroid bentuk':
    rot_origin = tuple(centroid)
elif rot_origin_choice == '(0,0)':
    rot_origin = (0.0, 0.0)
else:
    rot_origin = (rot_ox if rot_ox is not None else 0.0, rot_oy if rot_oy is not None else 0.0)

if scale_origin_choice == 'Centroid bentuk':
    scale_origin = tuple(centroid)
elif scale_origin_choice == '(0,0)':
    scale_origin = (0.0, 0.0)
else:
    scale_origin = (scale_ox if scale_ox is not None else 0.0, scale_oy if scale_oy is not None else 0.0)

# Apply transformations
matrices = []
intermediate_points = []
labels = []
current_pts = pts.copy()

if translate_on:
    current_pts, M = apply_translation(current_pts, tx, ty)
    matrices.append(M)
    intermediate_points.append(current_pts.copy())
    labels.append(f'Translasi tx={tx}, ty={ty}')

if rotate_on:
    current_pts, M = apply_rotation(current_pts, angle, origin=rot_origin)
    matrices.append(M)
    intermediate_points.append(current_pts.copy())
    labels.append(f'Rotasi {angle}° pusat={rot_origin}')

if scale_on:
    current_pts, M = apply_scaling(current_pts, sx, sy, origin=scale_origin)
    matrices.append(M)
    intermediate_points.append(current_pts.copy())
    labels.append(f'Dilatasi sx={sx}, sy={sy} pusat={scale_origin}')

if reflect_on:
    current_pts, M = apply_reflection(current_pts, axis=reflect_axis)
    matrices.append(M)
    intermediate_points.append(current_pts.copy())
    labels.append(f'Refleksi sumbu={reflect_axis}')

final_pts = current_pts

# Show main plot
with col1:
    st.subheader('Visualisasi')
    if step_by_step and intermediate_points:
        step = st.slider('Langkah transformasi', 0, len(intermediate_points), len(intermediate_points))
        if step == 0:
            fig = plot_shapes(pts, pts, title='Bentuk awal (belum ada transformasi)')
        else:
            fig = plot_shapes(pts, intermediate_points[step-1], title=f'Langkah {step}: {labels[step-1]}')
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = plot_shapes(pts, final_pts, title='Hasil transformasi')
        st.plotly_chart(fig, use_container_width=True)

    st.caption('Klik kanan pada gambar untuk menyimpan. Gunakan slider "Langkah transformasi" untuk melihat proses jika aktif.')

# Show matrices and coordinates
with col2:
    st.subheader('Matriks Transformasi & Koordinat')
    if combine:
        M_combined = chain_matrices(matrices[::-1]) if matrices else np.eye(3)
        st.markdown('**Matriks gabungan (komposisi dari semua transformasi)**')
        st.write(M_combined)
    else:
        st.info('Centang "Gabungkan..." untuk melihat matriks komposisi (gabungan).')

    st.markdown('---')
    st.write('**Koordinat titik (original)**')
    df_orig = df_from_points('orig', pts)
    st.dataframe(df_orig)

    st.write('**Koordinat titik (hasil)**')
    df_final = df_from_points('trans', final_pts)
    st.dataframe(df_final)

    if export_csv:
        csv_bytes = points_to_csv_bytes(pd.concat([df_orig.assign(type='original'), df_final.assign(type='transformed')], ignore_index=True))
        st.download_button('Unduh CSV koordinat', data=csv_bytes, file_name='koordinat_transformasi.csv', mime='text/csv')

    st.markdown('---')
    st.subheader('Contoh Pertanyaan / Aktivitas')
    st.markdown('- Apa yang berubah jika sx<0 pada dilatasi? (apa artinya tanda negatif)')
    st.markdown('- Jika rotasi 90° berpusat di origin, apa koordinat (x,y) menjadi apa?')
    st.markdown('- Bandingkan hasil komposisi Rotasi lalu Translasi vs Translasi lalu Rotasi')

# Footer: deployment instructions
st.markdown('---')
st.header('Cara menjalankan dan deploy ke GitHub + Streamlit Cloud')
st.markdown('''
1. Simpan file ini sebagai `virtual_lab_transformasi_geometri_streamlit.py`.
2. Buat virtual environment dan install dependencies:

```
python -m venv venv
source venv/bin/activate  # (Windows: venv\\Scripts\\activate)
pip install streamlit numpy pandas plotly
```

3. Jalankan secara lokal:

```
streamlit run virtual_lab_transformasi_geometri_streamlit.py
```

4. Deploy di GitHub:
   - Buat repository baru, tambahkan file, commit, dan push.
   - Di Streamlit Community Cloud (https://share.streamlit.io), hubungkan akun GitHub Anda, pilih repo & branch, lalu deploy.
''')

st.caption('Butuh penyesuaian konten pembelajaran, latihan soal otomatis, atau versi Bahasa Inggris? Katakan saja—saya bisa tambahkan!')
