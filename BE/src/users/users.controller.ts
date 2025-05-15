import {
  Body,
  Controller,
  Get,
  Param,
  Patch,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { CurrentUser } from 'src/auth/@decorator/current-user.decorator';
import { Roles } from 'src/auth/@decorator/roles.decorator';
import { ROLE } from 'src/auth/util/@enum/role.enum';
import { CloudinaryService } from 'src/file/file.service';
import { GetUserByKeywordDTO } from './dto/get-user.dto';
import { UpdateUserDto } from './dto/update-user.dto';
import { UsersService } from './users.service';

class EditUserNamesDto {
  firstName?: string;
  lastName?: string;
}

class ChangeUserActiveDto {
  isActive: boolean;
}

@Controller('users')
export class UsersController {
  constructor(
    private usersService: UsersService,
    private cloudinaryService: CloudinaryService,
  ) {}

  @Get('/doctor/:doctorId')
  findAll(@Param('doctorId') doctorId: string) {
    return this.usersService.findDoctorByID(doctorId);
  }

  @Get('/doctors')
  findAllDoctors(@CurrentUser() currentUser: any) {
    return this.usersService.findAllDoctors();
  }

  @Get('/keyword')
  async findUserByKeyword(@Body() keyword: GetUserByKeywordDTO) {
    return await this.usersService.findUserByKeyword(keyword);
  }

  @Patch('/edit-profile')
  async editProfile(
    @CurrentUser('userId') userId: string,
    @Body() updateUserDto: UpdateUserDto,
  ) {
    return await this.usersService.editProfile({ ...updateUserDto }, userId);
  }

  @Patch('change/avatar')
  @UseInterceptors(FileInterceptor('file'))
  async updateAvatar(
    @CurrentUser('userId') userId: string,
    @UploadedFile() file: Express.Multer.File,
  ) {
    const uploadedFile = await this.cloudinaryService.uploadFile(file);
    return await this.usersService.updateAvatar(userId, uploadedFile.url);
  }

  @Patch('/admin/edit-names/:userId')
  @Roles(ROLE.ADMIN)
  async editUserNames(
    @Param('userId') userId: string,
    @Body() dto: EditUserNamesDto,
  ) {
    console.log('editUserNames', userId, dto);
    return this.usersService.editUserNames(userId, dto);
  }

  @Patch('/admin/change-active/:userId')
  @Roles(ROLE.ADMIN)
  async changeUserActive(
    @Param('userId') userId: string,
    @Body() dto: ChangeUserActiveDto,
  ) {
    console.log('changeUserActive', userId, dto);
    return this.usersService.changeUserActive(userId, dto.isActive);
  }
}
