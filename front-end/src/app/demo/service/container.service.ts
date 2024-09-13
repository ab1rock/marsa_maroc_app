
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Container {
  code: string;
  date_time: string;
  detection_threshold: number;
  id: number;
  image_path: string;
}


@Injectable({
  providedIn: 'root'
})

export class ContainerService {

  private apiUrl = 'http://127.0.0.1:8000/api/containers/';  // URL de l'API Django
  private imageurl = 'http://127.0.0.1:8000/api/upload/';
  private vidUrl = 'http://127.0.0.1:8000/api/latest_data/';
  constructor(private http: HttpClient) { }

  getContainers(): Observable<any> { 
    return this.http.get<Container>(this.apiUrl);
  }

  addContainer(container: any): Observable<any> {
    return this.http.post(this.apiUrl, container);
  }

  updateContainer(container: any): Observable<any> {
    return this.http.put(`${this.apiUrl}${container.id}/`,container);
  }

  deleteContainer(id: number): Observable<any> {
    return this.http.delete(`${this.apiUrl}${id}/`);
  }
  uploadContainerImage(formData: FormData): Observable<any> {
    return this.http.post(this.imageurl, formData);
  }
  getLatestData(): Observable<any> {
    return this.http.get<any>(this.vidUrl);
  }

}