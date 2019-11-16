import argparse
import os
import shutil

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation study on Image Retrieval Model')
    parser.add_argument('--better_data_base',
                        default='car_uncropped_fixed_unrandom_layer1_resnet18_48_12_data_base.pth',
                        type=str, help='better database')
    parser.add_argument('--worse_data_base',
                        default='car_uncropped_random_unrandom_layer1_resnet18_48_12_data_base.pth',
                        type=str, help='worse database')
    parser.add_argument('--data_type', default='test', type=str, choices=['train', 'test'],
                        help='compared database type')
    parser.add_argument('--retrieval_num', default=8, type=int, help='retrieval number')
    parser.add_argument('--save_results', action='store_true', help='with save results or not')

    opt = parser.parse_args()

    BETTER_DATA_BASE, WORSE_DATA_BASE, RETRIEVAL_NUM = opt.better_data_base, opt.worse_data_base, opt.retrieval_num
    BETTER_DATA_TYPE, WORSE_DATA_TYPE = BETTER_DATA_BASE.split('_')[:2], WORSE_DATA_BASE.split('_')[:2]
    DATA_TYPE, SAVE_RESULTS = opt.data_type, opt.save_results
    if BETTER_DATA_TYPE != WORSE_DATA_TYPE:
        raise NotImplementedError('make sure the data type and crop type are same')

    better_data_base = torch.load('results/{}'.format(BETTER_DATA_BASE))
    worse_data_base = torch.load('results/{}'.format(WORSE_DATA_BASE))

    assert better_data_base['{}_images'.format(DATA_TYPE)] == worse_data_base['{}_images'.format(DATA_TYPE)]
    assert better_data_base['{}_labels'.format(DATA_TYPE)] == worse_data_base['{}_labels'.format(DATA_TYPE)]
    assert len(better_data_base['{}_features'.format(DATA_TYPE)]) == len(
        worse_data_base['{}_features'.format(DATA_TYPE)])
    if DATA_TYPE != 'train':
        assert better_data_base['gallery_images'] == worse_data_base['gallery_images']
        assert better_data_base['gallery_labels'] == worse_data_base['gallery_labels']
        assert len(better_data_base['gallery_features']) == len(worse_data_base['gallery_features'])
    num_query_images, better_num, equal_num = len(better_data_base['{}_images'.format(DATA_TYPE)]), 0, 0
    query_images = better_data_base['{}_images'.format(DATA_TYPE)]
    query_labels = better_data_base['{}_labels'.format(DATA_TYPE)]
    gallery_images = better_data_base['{}_images'.format('train' if DATA_TYPE == 'train' else 'gallery')]
    gallery_labels = torch.tensor(better_data_base['{}_labels'.format('train' if DATA_TYPE == 'train' else 'gallery')])

    # define a new metric to better evaluate the retrieval performance (Average Recall---AR)
    # when the retried image has the same label as query image, then we count the number of this case
    # finally we average the number, divide the retrieval num
    better_correct_all, worse_correct_all, better_correct_instance, worse_correct_instance = 0, 0, 0, 0

    progress_bar = tqdm(range(num_query_images), 'processing data...')
    for query_index in progress_bar:
        query_label = torch.tensor(query_labels[query_index])

        better_query_feature = better_data_base['{}_features'.format(DATA_TYPE)][query_index]
        better_query_feature = better_query_feature.view(1, *better_query_feature.size()).permute(1, 0, 2).contiguous()
        worse_query_feature = worse_data_base['{}_features'.format(DATA_TYPE)][query_index]
        worse_query_feature = worse_query_feature.view(1, *worse_query_feature.size()).permute(1, 0, 2).contiguous()

        better_gallery_features = better_data_base['{}_features'.format('train' if DATA_TYPE == 'train' else 'gallery')]
        better_gallery_features = better_gallery_features.permute(1, 2, 0).contiguous()
        worse_gallery_features = worse_data_base['{}_features'.format('train' if DATA_TYPE == 'train' else 'gallery')]
        worse_gallery_features = worse_gallery_features.permute(1, 2, 0).contiguous()

        better_sim_matrix = better_query_feature.bmm(better_gallery_features).mean(dim=0).squeeze(dim=0)
        worse_sim_matrix = worse_query_feature.bmm(worse_gallery_features).mean(dim=0).squeeze(dim=0)
        if (BETTER_DATA_TYPE[0] is not 'isc') or (BETTER_DATA_TYPE[0] is 'isc' and DATA_TYPE is 'train'):
            better_sim_matrix[query_index], worse_sim_matrix[query_index] = -1, -1
        better_idx = better_sim_matrix.argsort(dim=-1, descending=True)
        worse_idx = worse_sim_matrix.argsort(dim=-1, descending=True)

        better_correct_num = 0
        for index in better_idx[:RETRIEVAL_NUM]:
            retrieval_label = gallery_labels[index.item()]
            retrieval_status = (retrieval_label == query_label).item()
            if retrieval_status:
                better_correct_num += 1
        if better_correct_num > 0:
            better_correct_all += better_correct_num
            better_correct_instance += 1

        worse_correct_num = 0
        for index in worse_idx[:RETRIEVAL_NUM]:
            retrieval_label = gallery_labels[index.item()]
            retrieval_status = (retrieval_label == query_label).item()
            if retrieval_status:
                worse_correct_num += 1
        if worse_correct_num > 0:
            worse_correct_all += worse_correct_num
            worse_correct_instance += 1

        if better_correct_num > worse_correct_num:
            better_num += 1
        elif better_correct_num == worse_correct_num:
            equal_num += 1

        # save the results
        if better_correct_num > worse_correct_num and SAVE_RESULTS:
            base_path = '{}----{}'.format(BETTER_DATA_BASE.split('_data_base.')[0],
                                          WORSE_DATA_BASE.split('_data_base.')[0])
            if not os.path.exists('results/{}'.format(base_path)):
                os.mkdir('results/{}'.format(base_path))
            query_img_name = query_images[query_index]
            progress_bar.set_description('[{}/{}] saving results for better case: {}'
                                         .format(query_index + 1, num_query_images, query_img_name.split('/')[-1]))
            result_path = 'results/{}/{}'.format(base_path, query_img_name.split('/')[-1].split('.')[0])
            if os.path.exists(result_path):
                shutil.rmtree(result_path)
            os.mkdir(result_path)

            query_image = Image.open(query_img_name).convert('RGB').resize((256, 256), resample=Image.BILINEAR)
            query_image.save('{}/query_img.jpg'.format(result_path))
            os.mkdir('{}/better'.format(result_path))
            for num, index in enumerate(better_idx[:RETRIEVAL_NUM]):
                retrieval_image = Image.open(gallery_images[index.item()]).convert('RGB') \
                    .resize((256, 256), resample=Image.BILINEAR)
                draw = ImageDraw.Draw(retrieval_image)
                retrieval_label = gallery_labels[index.item()]
                retrieval_status = (retrieval_label == query_label).item()
                retrieval_prob = better_sim_matrix[index.item()].item()
                if retrieval_status:
                    draw.rectangle((0, 0, 255, 255), outline='green', width=5)
                else:
                    draw.rectangle((0, 0, 255, 255), outline='red', width=5)
                retrieval_image.save(
                    '{}/better/retrieval_img_{}_{}.jpg'.format(result_path, num + 1, '%.4f' % retrieval_prob))

            os.mkdir('{}/worse'.format(result_path))
            for num, index in enumerate(worse_idx[:RETRIEVAL_NUM]):
                retrieval_image = Image.open(gallery_images[index.item()]).convert('RGB') \
                    .resize((256, 256), resample=Image.BILINEAR)
                draw = ImageDraw.Draw(retrieval_image)
                retrieval_label = gallery_labels[index.item()]
                retrieval_status = (retrieval_label == query_label).item()
                retrieval_prob = worse_sim_matrix[index.item()].item()
                if retrieval_status:
                    draw.rectangle((0, 0, 255, 255), outline='green', width=8)
                else:
                    draw.rectangle((0, 0, 255, 255), outline='red', width=8)
                retrieval_image.save(
                    '{}/worse/retrieval_img_{}_{}.jpg'.format(result_path, num + 1, '%.4f' % retrieval_prob))

    print('better data base AR@{}:{:.2f}%----worse data base AR@{}:{:.2f}%, GE Rate@{}:{:.2f}%, Better Ratio@{}:{:.2f}'
          .format(RETRIEVAL_NUM, better_correct_all / (better_correct_instance * RETRIEVAL_NUM) * 100,
                  RETRIEVAL_NUM, worse_correct_all / (worse_correct_instance * RETRIEVAL_NUM) * 100,
                  RETRIEVAL_NUM, (better_num + equal_num) / num_query_images * 100,
                  RETRIEVAL_NUM, better_num / (num_query_images - equal_num - better_num)))
