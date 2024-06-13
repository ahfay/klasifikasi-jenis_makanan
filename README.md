# Tugas Pengenalan Pola - Klasifikasi Makanan

# Deskripsi Data
Dataset yang digunakan bernama "Food Classification" yang diambil dari platform Kaggle . Berisi kumpulan gambar yang menampilkan berbagai jenis makanan yang dibagi menjadi tiga kategori: Makanan, Makanan Penutup, dan Minuman. Setiap kategori mewakili jenis makanan berbeda yang biasa ditemukan di berbagai masakan. Jumlah total gambar sebanyak 426 gambar yang berformat JPG.

# Deskripsi Kategori

1. Meals<br>
Makanan ini mencakup berbagai hidangan utama yang biasa dikonsumsi saat sarapan, makan siang, atau makan malam. Hidangan ini sering terdiri dari beberapa bahan dan dapat mencakup campuran daging, sayuran, biji-bijian, dan saus.
2. Drinks<br>
 Minuman termasuk berbagai minuman seperti soda, minuman panas, air, dll dalam gelas berbentuk berbeda.
3. Desserts<br>
 Berbagai macam makanan yang dipanggang, camilan beku, juga dalam berbagai warna dan bentuk.

# Langkah-langkah



*   Dataset yang diunduh masih berformat zip, perlu diekstrak terlebih dahulu supaya bisa diakses isinya.
*   Setelah diekstrak, data sudah disimpan dalam folder-folder sesuai dengan kategorinya. Ada 3 folder yaitu meals yang berisi 212 gambar, drinks berisi 98 gambar, dan desserts berisi 116 gambar.
*   Data di masing-masing kategori diaugmentasi untuk diperbanyak variasinya, ada 6 proses augmentasi yaitu random rotate, random flip, random rotation, random shear, random crop. Hasil dari augmentasi menambahkan data citra di masing-masing kategori sebanyak 500 citra. Proses ini menggunakan code yang ada di data/data.py
*   Melakukan perubahan ukuran setiap citra menjadi ke ukuran 200 x 200 piksel, lalu mengubah format warna dari BGR ke RGB dan menormalisasi nilai pikselnya.
*   Mengubah kategeori menjadi numerik, kategori desserts menjadi 0, drink menjadi 1, dan meal menjadi 2.
*   Dilakukan pembagian yang akan digunakan 70% untuk training dan 30% testing.
*   Membuat model CNN dengan arsitektur seperti yang ada di file model/model.py
*   Model di compile dengan algoritma adam untuk optimasi dan sparse categorical crossectropy untuk fungsi loss.
*   Melakukan proses training model dengan parameter epochs 25, batch size 32, dan validation split sebesar 0.1.
*   Menyimpan model yang telah dilatih di folder model/food_classifier.keras
*   Melakukan evaluasi model dengan data testing dan mendapatkan skor akurasi 65.7% dan loss sebesar 0.722.
